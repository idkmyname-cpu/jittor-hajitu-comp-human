import jittor as jt
import numpy as np
import os
import argparse
import time
import random
import math

from jittor import nn
from jittor import optim

class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.current_epoch = 0
    
    def step(self):
        lr = self.eta_min + (self.base_lr - self.eta_min) * \
             (1 + math.cos(math.pi * self.current_epoch / self.T_max)) / 2
        self.optimizer.lr = lr
        self.current_epoch += 1
    
    def get_last_lr(self):
        return [self.optimizer.lr]

class ReduceLROnPlateau:
    """简化版的自适应学习率调度器"""
    def __init__(self, optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, threshold=0.001):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.best = None
        self.num_bad_epochs = 0
        
    def step(self, metrics):
        if self.best is None:
            self.best = metrics
        elif self.is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
            
    def is_better(self, a, best):
        if self.mode == 'min':
            return a < best - self.threshold
        else:
            return a > best + self.threshold
            
    def _reduce_lr(self):
        new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
        if new_lr != self.optimizer.lr:
            print(f"Reducing learning rate to {new_lr:.8f}")
            self.optimizer.lr = new_lr

from dataset.dataset import get_dataloader, transform
from dataset.format import id_to_name
from dataset.sampler import SamplerMix
from models.skin import create_model
# from models.zmh_skin import create_model

from dataset.exporter import Exporter

# Set Jittor flags
jt.flags.use_cuda = 1


alpha = 0.3    # MSE权重
beta = 0.5     # L1权重
gamma = 0.2    # 解剖学邻近损失权重
delta = 0.1    # 求和约束权重
k = 0.2    # 熵损失权重

def improved_skin_loss(outputs, targets, vertices, joints):
    """
    改进的皮肤权重损失函数
    包含平滑性约束和稀疏性约束，以及解剖学约束
    """
    # 基础损失
    mse_loss = nn.MSELoss()(outputs, targets)
    l1_loss = nn.L1Loss()(outputs, targets)

   
    # 解剖学邻近损失
    vertices_exp = vertices.unsqueeze(2)  # [B, N, 1, 3]
    joints_exp = joints.unsqueeze(1)      # [B, 1, J, 3]
    distances = jt.norm(vertices_exp - joints_exp, p=2, dim=-1)  # [B, N, J]
    closest_joint = jt.argmin(distances, dim=-1)[0]  # [B, N]
    closest_joint_value = jt.gather(outputs, dim=2, index=closest_joint.unsqueeze(-1)).squeeze(-1)
    ones_matrix = jt.ones_like(closest_joint_value)
    proximity_loss = ((closest_joint_value - ones_matrix) ** 2).mean()

    # 计算输出的熵，做稀疏性约束
    epsilon = 1e-8
    entropy = -jt.sum(outputs * (jt.log(outputs + epsilon) - jt.log(targets + epsilon)), dim=-1).mean()

    # 总损失
    total_loss = alpha * mse_loss + \
                 beta * l1_loss + \
                 gamma * proximity_loss + \
                 k * abs(entropy)

    return total_loss, mse_loss, l1_loss, proximity_loss, entropy

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
    )
    
    # Load pre-trained model if specified
    if args.pretrained_model:
        log_message(f"Loading pretrained model from {args.pretrained_model}")
        model.load(args.pretrained_model)
    
    # Create optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # 学习率调度器
    if args.use_scheduler:
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs,
            eta_min=args.learning_rate * 0.01
        )
    else:
        scheduler = None

    # Create dataloaders
    train_loader = get_dataloader(
        data_root=args.data_root,
        data_list=args.train_data_list,
        train=True,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=SamplerMix(num_samples=1024, vertex_samples=512),
        transform=transform,
        #num_workers=10,           # 10个并行进程
    )
    
    if args.val_data_list:
        val_loader = get_dataloader(
            data_root=args.data_root,
            data_list=args.val_data_list,
            train=False,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=SamplerMix(num_samples=1024, vertex_samples=512),
            transform=transform,
            #num_workers=4,           # 验证时减少worker数量
        )
    else:
        val_loader = None
    
    # Training loop
    best_loss = 99999999
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        
        train_loss_total = 0.0
        train_loss_mse = 0.0
        train_loss_l1 = 0.0
        train_loss_sum = 0.0
        train_loss_proximity = 0.0
        train_loss_entropy = 0.0

        start_time = time.time()
        for batch_idx, data in enumerate(train_loader):
            # Get data and labels
            vertices, joints, skin = data['vertices'], data['joints'], data['skin']

            vertices: jt.Var
            joints: jt.Var
            skin: jt.Var

            # Forward pass
            outputs = model(vertices, joints)

            total_loss, mse_loss, l1_loss, proximity_loss, entropy = improved_skin_loss(outputs, skin, vertices, joints)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            optimizer.backward(total_loss)
            optimizer.step()

            # Calculate statistics
            train_loss_total += total_loss.item()
            train_loss_mse += mse_loss.item()
            train_loss_l1 += l1_loss.item()
            #train_loss_sum += sum_loss.item()
            train_loss_proximity += proximity_loss.item()
            train_loss_entropy += entropy.item()
            
            # Print progress
            if (batch_idx + 1) % args.print_freq == 0 or (batch_idx + 1) == len(train_loader):
                log_message(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                           f"Total: {total_loss.item():.4f} MSE: {mse_loss.item():.4f} "
                           f"L1: {l1_loss.item():.4f} "
                           f"Prox: {proximity_loss.item():.4f} Entropy: {entropy.item():.4f}")
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()

        train_loss_total /= len(train_loader)
        train_loss_mse /= len(train_loader)
        train_loss_l1 /= len(train_loader)
        #train_loss_sum /= len(train_loader)
        train_loss_proximity /= len(train_loader)
        train_loss_entropy /= len(train_loader)
        epoch_time = time.time() - start_time
        
        current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.lr
        log_message(f"Epoch [{epoch+1}/{args.epochs}] "
                   f"Total: {train_loss_total:.4f} MSE: {train_loss_mse:.4f} "
                   f"L1: {train_loss_l1:.4f}"
                   f"Prox: {train_loss_proximity:.4f} Entropy: {train_loss_entropy:.4f} "
                   f"Time: {epoch_time:.2f}s LR: {current_lr:.6f}")

        # Validation phase
        if val_loader is not None and (epoch + 1) % args.val_freq == 0:
            model.eval()
            val_loss_total = 0.0
            val_loss_mse = 0.0
            val_loss_l1 = 0.0
            #val_loss_sum = 0.0
            val_loss_proximity = 0.0
            val_loss_entropy = 0.0
            
            show_id = np.random.randint(0, len(val_loader))
            for batch_idx, data in enumerate(val_loader):
                # Get data and labels
                vertices, joints, skin = data['vertices'], data['joints'], data['skin']
                
                # Forward pass
                outputs = model(vertices, joints)
                total_loss, mse_loss, l1_loss,  proximity_loss, entropy = improved_skin_loss(outputs, skin, vertices, joints)
                
                # export render results(which is slow, so you can turn it off)
                if batch_idx == show_id:
                    exporter = Exporter()
                    for i in id_to_name:
                        name = id_to_name[i]
                        # export every joint's corresponding skinning
                        exporter._render_skin(path=f"tmp/skin/epoch_{epoch}/{name}_ref.png",vertices=vertices.numpy()[0], skin=skin.numpy()[0, :, i], joint=joints[0, i])
                        exporter._render_skin(path=f"tmp/skin/epoch_{epoch}/{name}_pred.png",vertices=vertices.numpy()[0], skin=outputs.numpy()[0, :, i], joint=joints[0, i])

                val_loss_total += total_loss.item()
                val_loss_mse += mse_loss.item()
                val_loss_l1 += l1_loss.item()
                #val_loss_sum += sum_loss.item()
                val_loss_proximity += proximity_loss.item()
                val_loss_entropy += entropy.item()
            
            # Calculate validation statistics
            val_loss_total /= len(val_loader)
            val_loss_mse /= len(val_loader)
            val_loss_l1 /= len(val_loader)
            #val_loss_sum /= len(val_loader)
            val_loss_proximity /= len(val_loader)
            val_loss_entropy /= len(val_loader)
            
            log_message(f"Validation - Total: {val_loss_total:.4f} MSE: {val_loss_mse:.4f} L1: {val_loss_l1:.4f} "
                       f"Prox: {val_loss_proximity:.4f} Entropy: {val_loss_entropy:.4f}")
            
            # Save best model
            if val_loss_l1 < best_loss:
                best_loss = val_loss_l1
                model_path = os.path.join(args.output_dir, 'best_model.pkl')
                model.save(model_path)
                log_message(f"Saved best model with l1 loss {best_loss:.4f} to {model_path}")
        
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
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam'],
                        help='Optimizer to use')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 penalty)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for SGD optimizer')
    
    # Enhanced loss parameters
    parser.add_argument('--use_enhanced_loss', action='store_true',
                        help='Use enhanced loss with smoothness and sum constraints')

    # Learning rate scheduling
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use cosine annealing learning rate scheduler')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output/skin',
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