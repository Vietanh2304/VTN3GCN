# analyze_failures.py
import torch
import numpy as np
import json
from collections import defaultdict

def extract_failure_cases(model, dataloader, class_names, device, top_k=20):
    model.eval()
    failures = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # inputs là dict: {'clip': ..., 'poseflow': ...}
            batch = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)
            
            output = model(**batch)
            logits = output["logits"]
            probs = torch.softmax(logits, dim=-1)
            pred_class = probs.argmax(dim=-1)
            pred_conf = probs.max(dim=-1).values
            
            for i in range(len(labels)):
                gt = labels[i].item()
                pred = pred_class[i].item()
                conf = pred_conf[i].item()
                
                if gt != pred:
                    failures.append({
                        'batch_idx': batch_idx,
                        'sample_idx': batch_idx * dataloader.batch_size + i,
                        'gt_class': gt,
                        'gt_name': class_names[gt],
                        'pred_class': pred,
                        'pred_name': class_names[pred],
                        'confidence': conf,
                        'top3': [
                            (class_names[idx], probs[i][idx].item())
                            for idx in probs[i].topk(3).indices.tolist()
                        ]
                    })
    
    failures.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"Total failures: {len(failures)}")
    print(f"\nTop-{top_k} high-confidence failures:")
    for f in failures[:top_k]:
        print(f"  GT: {f['gt_name']:20s} | Pred: {f['pred_name']:20s} | Conf: {f['confidence']:.3f}")
    
    return failures


def analyze_confusion_pairs(failures, top_n=10):
    from collections import defaultdict
    pair_count = defaultdict(int)
    for f in failures:
        pair = (f['gt_name'], f['pred_name'])
        pair_count[pair] += 1
    
    sorted_pairs = sorted(pair_count.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop-{top_n} most confused pairs (GT → Predicted):")
    for (gt, pred), count in sorted_pairs[:top_n]:
        print(f"  {gt:20s} → {pred:20s} : {count} times")
    
    return sorted_pairs