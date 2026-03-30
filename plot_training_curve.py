import matplotlib.pyplot as plt
import re

# Read the training log directly
with open('baseline_150epochs_train.txt', 'r') as f:
    content = f.read()

# Parse epoch, avg_loss, val_iou
pattern = r'\[epoch: (\d+)/\d+\]\navg_loss: ([\d.]+)\nval_IoU: ([\d.]+)'
matches = re.findall(pattern, content)

epochs = [int(m[0]) for m in matches]
avg_loss = [float(m[1]) for m in matches]
val_iou = [float(m[2]) for m in matches]

print(f"Parsed {len(epochs)} epochs, {len(avg_loss)} loss values, {len(val_iou)} IoU values")

fig, ax1 = plt.subplots(figsize=(10, 6))

color_loss = '#d62728'
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', color=color_loss, fontsize=12)
ax1.plot(epochs, avg_loss, color=color_loss, linewidth=1.5, label='Loss')
ax1.tick_params(axis='y', labelcolor=color_loss)
ax1.set_xlim(1, max(epochs))

ax2 = ax1.twinx()
color_iou = '#2ca02c'
ax2.set_ylabel('Val IoU', color=color_iou, fontsize=12)
ax2.plot(epochs, val_iou, color=color_iou, linewidth=1.5, label='Val IoU')
ax2.tick_params(axis='y', labelcolor=color_iou)

# mark best IoU
best_idx = val_iou.index(max(val_iou))
best_epoch = epochs[best_idx]
best_iou = val_iou[best_idx]
ax2.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7)
ax2.annotate(f'Best: {best_iou:.4f}\n(epoch {best_epoch})',
             xy=(best_epoch, best_iou),
             xytext=(best_epoch + 10, best_iou - 0.02),
             fontsize=9,
             arrowprops=dict(arrowstyle='->', color='gray'))

fig.tight_layout()
plt.title('Baseline 150 Epochs: Loss & Val IoU', fontsize=14)
plt.savefig('baseline_150_training_curve.png', dpi=150, bbox_inches='tight')
plt.savefig('baseline_150_training_curve.pdf', bbox_inches='tight')
plt.show()
print(f"Best IoU: {best_iou:.4f} @ epoch {best_epoch}")
