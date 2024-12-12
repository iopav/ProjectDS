import torch
from sklearn.metrics import roc_auc_score
from PIL import Image
from refer.utils import load_checkpoint, save_checkpoint, latest_checkpoint_path




def train_epoch(model, optimizer, scheduler, loss_fn, train_loader, device):
    
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        running_acc += (outputs.argmax(1) == labels).float().mean().item()
        l = loss_fn(outputs, labels)
        l.backward()
        optimizer.step()
        running_loss += l.item()
    running_loss /= len(train_loader)
    running_acc  /= len(train_loader)
    if scheduler is not None:
        scheduler.step()
        learning_rate = scheduler.get_last_lr()[0]
    else:
        learning_rate = optimizer.param_groups[0]['lr']
    return running_loss, running_acc, learning_rate

def validate_epoch(model, val_loader, loss_fn, epoch, writer, device):
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            l = loss_fn(outputs, labels)
            val_loss += l.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(outputs[:, 1].cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total
    auc = roc_auc_score(true_labels, predictions)
    return val_loss, accuracy, auc
        
    
    
def infer(image_path, model, transform, checkpoint_dir, device):
    model, _, _, _ = load_checkpoint(checkpoint_dir, model)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    
    return predicted.item()