import jittor as jt
import numpy as np
import os
import argparse
import time
import random

from jittor import nn
from jittor import optim

from dataset.dataset import get_dataloader, transform
from dataset.sampler import SamplerMix
from dataset.exporter import Exporter
from models.skeleton import create_model

from models.metrics import J2J

# Set Jittor flags
jt.flags.use_cuda = 1


Alpha = 0.5  # 骨骼长度对称性损失的权重
Beta = 0.1   # 关节位置对称性损失的权重
Gamma = 0.1  # 相对位置损失的权重

def symmetric_length(outputs):
    """
    计算左右骨骼的长度

    Args:
        outputs (jt.Var): 模型的预测输出，形状为 (batch, 66)

    Returns:
        left_out (jt.Var): 左骨骼的长度
        right_out (jt.Var): 右骨骼的长度
    """
    l_bones = [[3,6], [6,7], [7,8], [8,9], [0,14], [14,15], [15,16], [16,17]]
    total_l_bones = jt.zeros((outputs.shape[0], 8), dtype=outputs.dtype)
    for i, bone in enumerate(l_bones):
        left_indices = [bone[0] * 3, bone[1] * 3]
        left_out = jt.norm(outputs[:, left_indices[0]:left_indices[0] + 3] - outputs[:, left_indices[1]:left_indices[1] + 3], dim=1)
        total_l_bones[:, i] = left_out

    r_bones = [[3,10], [10,11], [11,12], [12,13], [0,18], [18,19], [19,20], [20,21]]
    total_r_bones = jt.zeros((outputs.shape[0], 8), dtype=outputs.dtype)
    for i, bone in enumerate(r_bones):
        right_indices = [bone[0] * 3, bone[1] * 3]
        right_out = jt.norm(outputs[:, right_indices[0]:right_indices[0] + 3] - outputs[:, right_indices[1]:right_indices[1] + 3], dim=1)
        total_r_bones[:, i] = right_out

    return total_l_bones, total_r_bones

def symmetric_joint(outputs, targets):
    """
    计算左右关节位置

    Args:
        outputs (jt.Var): 模型的预测输出，形状为 (batch, 66)
        targets (jt.Var): 真实的关节标签，形状为 (batch, 66)

    Returns:
        left_outputs (jt.Var): 左关节的位置
        right_outputs (jt.Var): 右关节的位置
        left_targets (jt.Var): 左关节的真实位置
        right_targets (jt.Var): 右关节的真实位置
    """
    left_indices = jt.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17], dtype=jt.int32)
    right_indices = jt.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 18, 19, 20, 21], dtype=jt.int32)
    mid_indices = jt.array([0, 1, 2, 3, 4, 5], dtype=jt.int32)
    batch_size = outputs.shape[0]
    
    # 根据target确定轴
    left_hand  = targets[:, 27:30]
    right_hand = targets[:, 39:42]
    diff_sq = (left_hand - right_hand).sqr()
    axis_indices = jt.argmax(diff_sq, dim=1)[0]
    
    indices = mid_indices.reshape(1, -1) * 3 + axis_indices.reshape(-1, 1)
    mid_outputs = jt.gather(outputs, dim=1, index=indices)
    mean_values = jt.mean(mid_outputs, dim=1, keepdims=True)
    batch_indices = jt.arange(batch_size).reshape(-1, 1).expand(batch_size, 14)
    joint_indices = jt.arange(14).reshape(1, -1).expand(batch_size, 14)
        
    l_cols = jt.array([idx*3 + j for idx in left_indices for j in range(3)])
    left_outputs = jt.gather(outputs, dim=1, index=l_cols.reshape(1, -1).expand(batch_size, -1))
    left_targets = jt.gather(targets, dim=1, index=l_cols.reshape(1, -1).expand(batch_size, -1))
    left_outputs = left_outputs.reshape(batch_size, -1, 3)
    left_outputs[batch_indices, joint_indices, axis_indices.reshape(-1, 1)] = 2 * mean_values - left_outputs[batch_indices, joint_indices, axis_indices.reshape(-1, 1)]
    left_outputs = left_outputs.reshape(batch_size, -1)

    r_cols = jt.array([idx*3 + j for idx in right_indices for j in range(3)])
    right_outputs = jt.gather(outputs, dim=1, index=r_cols.reshape(1, -1).expand(batch_size, -1))
    right_targets = jt.gather(targets, dim=1, index=r_cols.reshape(1, -1).expand(batch_size, -1))
    right_outputs = right_outputs.reshape(batch_size, -1, 3)
    right_outputs[batch_indices, joint_indices, axis_indices.reshape(-1, 1)] = 2 * mean_values - right_outputs[batch_indices, joint_indices, axis_indices.reshape(-1, 1)]
    right_outputs = right_outputs.reshape(batch_size, -1)
    
    right_targets_left = right_targets.reshape(batch_size, -1, 3).clone()
    right_targets_left[batch_indices, joint_indices, axis_indices.reshape(-1, 1)] = 2 * mean_values - right_targets_left[batch_indices, joint_indices, axis_indices.reshape(-1, 1)]
    right_targets_left = right_targets_left.reshape(batch_size, -1)
    
    return left_outputs, right_outputs, left_targets, right_targets, right_targets_left


def composite_loss(outputs, targets):
    """
    计算包含骨骼对称性等先验知识的损失函数。

    Args:
        outputs (jt.Var): 模型的预测输出，形状为 (batch, 66)
        targets (jt.Var): 真实的关节标签，形状为 (batch, 66)

    Returns:
        jt.Var: 包含对称性的总损失
    """
    mse_loss = nn.MSELoss()(outputs, targets)
    
    # 骨长度左右对称
    total_l_bones, total_r_bones = symmetric_length(outputs)
    symmetric_loss = nn.MSELoss()(total_l_bones, total_r_bones)
    
    # 关节左右对称
    left_outputs, right_outputs, left_targets, right_targets, right_targets_left = symmetric_joint(outputs, targets)
    target_loss = nn.MSELoss()(left_outputs, right_targets_left)
    l_sym_joint_loss = nn.MSELoss()(left_outputs, right_targets)
    r_sym_joint_loss = nn.MSELoss()(right_outputs, left_targets)
    
    # 相对位置损失
    batch_size = outputs.shape[0]
    outputs_res = outputs.reshape(batch_size, 22, 3)
    outputs_exp1 = outputs_res.unsqueeze(2)     # [B, J, 1, 3]
    outputs_exp2 = outputs_res.unsqueeze(1)     # [B, 1, J, 3]
    diff_out = outputs_exp1 - outputs_exp2
    dist_matrix_out = jt.sqrt((diff_out ** 2).sum(-1))
    targets_res = targets.reshape(batch_size, 22, 3)
    targets_exp1 = targets_res.unsqueeze(2)     # [B, J, 1, 3]
    targets_exp2 = targets_res.unsqueeze(1)     # [B, 1, J, 3]
    diff_tar = targets_exp1 - targets_exp2
    dist_matrix_tar = jt.sqrt((diff_tar ** 2).sum(-1))
    relative_loss = nn.MSELoss()(dist_matrix_out, dist_matrix_tar)
    # 组合损失
    alpha = Alpha
    beta = Beta if target_loss < 10 else 0
    gamma = Gamma
    total_loss = (1-alpha-beta) * mse_loss + alpha * symmetric_loss + beta * (l_sym_joint_loss + r_sym_joint_loss) + gamma * relative_loss.item()
    return total_loss


def train(args):
    """
    Main training function
    
    Args:
        args: Command line arguments
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Set up logging
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    
    def log_message(message):
        """Helper function to log messages to file and print to console"""
        with open(log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)
    
    # Log training parameters
    log_message(f"Starting training with parameters: {args}")
    
    # Create model
    model = create_model(
        model_name=args.model_name,
        model_type=args.model_type
    )
    
    sampler = SamplerMix(num_samples=1024, vertex_samples=512)
    
    # Load pre-trained model if specified
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Create loss function
    # criterion = nn.MSELoss()#损失函数，可以看看换成什么或者怎么修改
    
    # Create dataloaders
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=sampler,
        transform=transform,
    )
    
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            transform=transform,
        )
    else:
        val_loader = None
    
    # Training loop
    best_loss = 99999999
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            vertices, joints = data['vertices'], data['joints']
            
            vertices = vertices.permute(0, 2, 1)  # [B, 3, N]

            outputs = model(vertices)
            joints = joints.reshape(outputs.shape[0], -1)
            loss = composite_loss(outputs, joints)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.step()
            
            # Calculate statistics
            train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Loss: {loss.item():.4f}")
        
        # Calculate epoch statistics
        train_loss /= len(train_loader)
        epoch_time = time.time() - start_time
        
        log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Train Loss: {train_loss:.4f} "
                   f"Time: {epoch_time:.2f}s "
                   f"LR: {optimizer.lr:.6f}")

        # Validation phase
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss = 0.0
            J2J_loss = 0.0
            
            show_id = np.random.randint(0, len(val_loader))
            for batch_idx, data in enumerate(val_loader):
                # Get data and labels
                vertices, joints = data['vertices'], data['joints']
                joints = joints.reshape(joints.shape[0], -1)
                
                # Reshape input if needed
                if vertices.ndim == 3:  # [B, N, 3]
                    vertices = vertices.permute(0, 2, 1)  # [B, 3, N]
                
                # Forward pass
                outputs = model(vertices)
                loss = composite_loss(outputs, joints)
                
                # export render results
                if batch_idx == show_id:
                    exporter = Exporter()
                    # export every joint's corresponding skinning
                    from dataset.format import parents
                    exporter._render_skeleton(path=f"tmp/skeleton/epoch_{epoch}/skeleton_ref.png", joints=joints[0].numpy().reshape(-1, 3), parents=parents)
                    exporter._render_skeleton(path=f"tmp/skeleton/epoch_{epoch}/skeleton_pred.png", joints=outputs[0].numpy().reshape(-1, 3), parents=parents)
                    exporter._render_pc(path=f"tmp/skeleton/epoch_{epoch}/vertices.png", vertices=vertices[0].permute(1, 0).numpy())

                val_loss += loss.item()
                for i in range(outputs.shape[0]):
                    J2J_loss += J2J(outputs[i].reshape(-1, 3), joints[i].reshape(-1, 3)).item() / outputs.shape[0]
            
            # Calculate validation statistics
            val_loss /= len(val_loader)
            J2J_loss /= len(val_loader)
            
            log_message(f"Validation Loss: {val_loss:.4f} J2J Loss: {J2J_loss:.4f}")
            
            # Save best model
            if J2J_loss < best_loss:
                best_loss = J2J_loss
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best model with loss {best_loss:.4f} to {model_path}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pkl')
            model.save(checkpoint_path)
            log_message(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pkl')
    model.save(final_model_path)
    log_message(f"Training completed. Saved final model to {final_model_path}")
    
    return model, best_loss

def main():
    """Parse arguments and start training"""
    parser = argparse.ArgumentParser(description='Train a point cloud model')
    
    # Dataset parameters
    parser.add_argument('--train_data_list', type=str, required=True,
                        help='Path to the training data list file')
    parser.add_argument('--val_data_list', type=str, default='',
                        help='Path to the validation data list file')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for the data files')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='pct',
                        choices=['pct', 'pct2', 'custom_pct', 'skeleton'],
                        help='Model architecture to use')
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'enhanced'],
                        help='Model type for skeleton model')
    parser.add_argument('--pretrained_model', type=str, default='',
                        help='Path to pretrained model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.00001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output/skeleton',
                        help='Directory to save output files')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='Print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save frequency')
    parser.add_argument('--val_freq', type=int, default=1,
                        help='Validation frequency')
    
    args = parser.parse_args()
    
    # Start training
    train(args)

def seed_all(seed):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    seed_all(123)
    main()