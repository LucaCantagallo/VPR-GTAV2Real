# logger.py
import json
import datetime
import os

class ExperimentLogger:
    def __init__(self, work_dir, n_epochs, batch_size, lr, dataset_name):
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dataset_name = dataset_name
        
        self.start_time = datetime.datetime.now()
        self.epoch_logs = []
        self.min_loss = float('inf')
        self.best_epoch = -1
        
    def log_epoch(self, epoch, train_loss, valid_loss):
        """Registra i valori di un'epoca e aggiorna la migliore."""
        self.epoch_logs.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "valid_loss": float(valid_loss)
        })
        
        if valid_loss < self.min_loss:
            self.min_loss = valid_loss
            self.best_epoch = epoch
            return True  # epoca corrente è la migliore
        return False
    
    def set_total_time(self, total_time):
        self.total_time = total_time
    
    def save_summary(self):
        """Salva JSON strutturato e un file txt leggibile."""
        end_time = datetime.datetime.now()
        total_seconds = (end_time - self.start_time).total_seconds()
        
        summary = {
            "start_time": str(self.start_time),
            "end_time": str(end_time),
            "total_seconds": total_seconds,
            "total_minutes": total_seconds / 60,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "dataset": self.dataset_name,
            "best_epoch": self.best_epoch,
            "best_validation_loss": self.min_loss,
            "epochs": self.epoch_logs
        }
        
        # Salva JSON
        json_path = os.path.join(self.work_dir, "summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        # Salva txt leggibile
        txt_path = os.path.join(self.work_dir, "summary.txt")
        with open(txt_path, "w") as f:
            f.write(f"Experiment Summary\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Batch size: {self.batch_size}, Learning rate: {self.lr}\n")
            f.write(f"Start time: {self.start_time}\n")
            f.write(f"End time: {end_time}\n")
            f.write(f"Total time (min): {total_seconds/60:.2f}\n")
            f.write(f"Number of epochs: {self.n_epochs}\n")
            f.write(f"Best epoch: {self.best_epoch}, Best validation loss: {self.min_loss:.6f}\n")
            f.write("\nPer-epoch results:\n")
            for ep in self.epoch_logs:
                f.write(f"Epoch {ep['epoch']}: train_loss={ep['train_loss']:.6f}, valid_loss={ep['valid_loss']:.6f}\n")
        
        print(f"Summary saved to:\n- {json_path}\n- {txt_path}")
