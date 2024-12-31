# train_utils.py

import torch
from backpack import backpack, extend
from backpack.extensions import BatchGrad, BatchL2Grad
# from adaptive_batch_size import compute_gradient_diversity
from utils import progress_bar
import torch.nn as nn
import csv
import os
import time

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def compute_grad_sum_norm(self, accumulated_grads):
        """Compute the norm of the sum of accumulated gradients."""
        flattened_grads = [grad.flatten() for grad in accumulated_grads]
        concatenated_grads = torch.cat(flattened_grads)
        return torch.norm(concatenated_grads).item() ** 2

    def train_epoch(self, dataloader, epoch):
        """Abstract method for training one epoch."""
        raise NotImplementedError("Subclasses must implement train_epoch!")


class SGDTrainer(Trainer):
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        epoch_start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        epoch_time = time.time() - epoch_start_time
        train_acc = 100. * correct / total
        return {
            "train_loss": train_loss / len(dataloader),
            "train_accuracy": train_acc,
            "epoch_time": epoch_time
        }


class DiveBatchTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, resize_freq, max_batch_size, min_batch_size=128, delta=0.02, dataset_size=50000):
        super().__init__(model, optimizer, criterion, device)
        self.resize_freq = resize_freq
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.last_grad_diversity = 1 # To store the last gradient diversity
        self.delta = delta  # Threshold delta for gradient diversity
        self.dataset_size = dataset_size
        # Ensure model and criterion are extended once
        # if not hasattr(model, 'backpack_extensions'):
        #     self.model = extend(model)
        if not hasattr(criterion, 'backpack_extensions'):
            self.criterion = extend(criterion)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        train_loss, correct, total = 0, 0, 0
        accumulated_grads = [torch.zeros_like(param.detach().cpu()) for param in self.model.parameters()]
        grad_square_sums =  None
        individual_grad_norm_sum = 0
        # individual_grad_norm_sum2 = 0
        self.current_batch_size = dataloader.batch_size
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            if (self.current_batch_size != self.max_batch_size) or (self.resize_freq != 0 and (epoch) % self.resize_freq == 0):
                set_bn_eval(self.model)
                with backpack(BatchL2Grad()):
                    loss.backward()
                set_bn_train(self.model)
                actual_batch_size = inputs.size(0)
                grad_square_sums = torch.zeros(actual_batch_size, device='cpu')
                
                for j, param in enumerate(self.model.parameters()):
                    accumulated_grads[j] += param.grad.detach().cpu()
                    individual_grad_norm_sum += (param.batch_l2).sum().item()
                    # grad_square_sums += param.batch_l2.view(param.batch_l2.size(0), -1).sum(dim=1).detach().cpu()
                # individual_grad_norm_sum2 += grad_square_sums.sum().item()
            else:
                loss.backward()

            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        # breakpoint()
        grad_sum_norm = self.compute_grad_sum_norm(accumulated_grads)
        # breakpoint()
        grad_diversity = self.compute_gradient_diversity(grad_sum_norm, individual_grad_norm_sum)
        self.last_grad_diversity = grad_diversity
        return {
            "train_loss": train_loss / len(dataloader),
            "train_accuracy": 100. * correct / total,
            "grad_diversity": grad_diversity
        }

    def compute_gradient_diversity(self, grad_sum_norm, individual_grad_norms):
        return individual_grad_norms /(grad_sum_norm + 1e-10)
    
    def resize_batch(self):
        """
        Adjust the current batch size based on the last gradient diversity.
        Returns the new batch size.
        """
        if self.last_grad_diversity is None:
            print("No gradient diversity data available for resizing.")
            return self.current_batch_size  # No change

        old_batch_size = self.current_batch_size
        new_gd = self.last_grad_diversity

        # Call the provided update_batch_size function
        new_batch_size = self.update_batch_size(old_batch_size, new_gd)

        if new_batch_size != old_batch_size:
            print(f"Resizing batch size from {old_batch_size} to {new_batch_size}")
            self.current_batch_size = new_batch_size
        else:
            print(f"Batch size remains at {self.current_batch_size}")
        
        return new_batch_size

    def update_batch_size(self, old_batch_size, new_gd):
        """
        Update batch size based on gradient diversity.
        """
        if not isinstance(new_gd, torch.Tensor):
            new_gd = torch.tensor(new_gd, device=self.device)
        if torch.isnan(new_gd).any() or torch.isinf(new_gd).any():
            print("Gradient diversity contains NaN or Inf. Setting batch size to max_batch_size.")
            return self.max_batch_size
        # Compute new batch size
        scaling_factor = self.delta * new_gd * self.dataset_size
        new_batch_size = int(min(max(scaling_factor, old_batch_size), self.max_batch_size))
        new_batch_size = max(new_batch_size, self.min_batch_size)  # Ensure not below min_batch_size
        return new_batch_size



def test(model, optimizer, scheduler, testloader, criterion, device, epoch, progress_bar, best_acc, checkpoint_dir, checkpoint_file):
    print('\nValidation...')
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Calculate accuracy
    acc = 100. * correct / total
    is_best = acc > best_acc
    best_acc = max(acc, best_acc)

    # Prepare state for saving
    state = {
        'epoch': epoch,  # Save the next epoch number
        'net': model.state_dict(),
        'best_acc': best_acc,
        'current_acc': acc,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'is_best': is_best
    }

    # Save the latest checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
    torch.save(state, latest_checkpoint_path)
    print(f"Saved latest checkpoint: {latest_checkpoint_path}")

    # Save the best checkpoint if this is the best accuracy
    if is_best:
        best_checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file.replace('.pth', '_best.pth'))
        torch.save(state, best_checkpoint_path)
        print(f"Updated best checkpoint: {best_checkpoint_path}")

    return test_loss / len(testloader), acc, best_acc



def log_metrics(log_file, epoch, train_loss, train_acc, val_loss, val_acc, lr, batch_size, epoch_time, eval_time, abs_time, memory_allocated, memory_reserved, grad_diversity=None):
    memory_allocated_mb = memory_allocated / (1024 * 1024)
    memory_reserved_mb = memory_reserved / (1024 * 1024)
    
    with open(log_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc',
            'learning_rate', 'batch_size', 'epoch_time', 'eval_time', 'abs_time', 'memory_allocated_mb', 'memory_reserved_mb', 'grad_diversity'
        ])
        writer.writerow({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': lr,
            'batch_size': batch_size,
            'epoch_time': epoch_time,  # Time taken for one epoch
            'eval_time': eval_time,  # Time taken for evaluation
            'abs_time': abs_time,  # Absolute time taken
            'memory_allocated_mb': round(memory_allocated_mb, 2),
            'memory_reserved_mb': round(memory_reserved_mb, 2),
            'grad_diversity': round(grad_diversity, 4) if grad_diversity is not None else None,
        })

    def resize_batch(self):
        """
        Adjust the current batch size based on the last gradient diversity.
        """
        if self.last_grad_diversity is None:
            print("No gradient diversity data available for resizing.")
            return self.current_batch_size  # No change

        adjustment_factor = 1.1  # Factor to adjust the batch size
        diversity_threshold_low = 0.5  # Below this, increase batch size
        diversity_threshold_high = 1.0  # Above this, decrease batch size

        new_batch_size = self.current_batch_size

        if self.last_grad_diversity < diversity_threshold_low:
            # Increase batch size
            new_batch_size = int(self.current_batch_size * adjustment_factor)
            new_batch_size = min(new_batch_size, self.max_batch_size)
            if new_batch_size != self.current_batch_size:
                print(f"Increasing batch size from {self.current_batch_size} to {new_batch_size} based on low grad diversity ({self.last_grad_diversity:.4f})")
                self.current_batch_size = new_batch_size
        elif self.last_grad_diversity > diversity_threshold_high:
            # Decrease batch size
            new_batch_size = int(self.current_batch_size / adjustment_factor)
            new_batch_size = max(new_batch_size, self.min_batch_size)
            if new_batch_size != self.current_batch_size:
                print(f"Decreasing batch size from {self.current_batch_size} to {new_batch_size} based on high grad diversity ({self.last_grad_diversity:.4f})")
                self.current_batch_size = new_batch_size
        else:
            print(f"No resizing needed. Current batch size remains at {self.current_batch_size} (grad diversity: {self.last_grad_diversity:.4f})")

        return self.current_batch_size



def set_bn_eval(model):
    """Set all BatchNorm layers in the model to eval mode."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

def set_bn_train(model):
    """Set all BatchNorm layers in the model to train mode."""
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()