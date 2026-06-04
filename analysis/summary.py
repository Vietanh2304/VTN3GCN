import os
import json
import glob
import statistics

# Tìm tất cả các file result.json trong các thư mục bắt đầu bằng VSL400_ hoặc MultiVSL_
result_files = glob.glob('VSL400_*/result.json')

if not result_files:
    print("Chưa tìm thấy file kết quả nào. Bạn đã chạy xong chưa?")
    exit()

# Dictionary để gom nhóm các lần chạy cùng phương pháp
experiments = {}

for file_path in result_files:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        views = "+".join(data.get('views', []))
        method = data.get('method', 'N/A')
        val_acc = data.get('best_val_acc', 0) * 100
        test_acc = data.get('test_acc', 0) * 100
        params = data.get('num_params', 0) / 1e6
        time_m = data.get('total_time_sec', 0) / 60
        
        # Tạo khóa (key) để gom nhóm. VD: "front+left+right | concat"
        key = f"{views} | {method}"
        
        if key not in experiments:
            experiments[key] = {'test_accs': [], 'params': params, 'times': []}
            
        experiments[key]['test_accs'].append(test_acc)
        experiments[key]['times'].append(time_m)
        
    except Exception as e:
        print(f"Lỗi khi đọc file {file_path}: {e}")

# Xử lý tính Mean và Std Dev (±)
results = []
for key, metrics in experiments.items():
    accs = metrics['test_accs']
    
    # Tính trung bình
    mean_acc = statistics.mean(accs)
    
    # Tính độ lệch chuẩn (±). Nếu chỉ chạy 1 lần thì độ lệch chuẩn = 0
    std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.00
    
    avg_time = statistics.mean(metrics['times'])
    num_runs = len(accs)
    
    results.append((key, mean_acc, std_acc, metrics['params'], avg_time, num_runs))

# Sắp xếp theo Accuracy giảm dần
results.sort(key=lambda x: x[1], reverse=True)

# In bảng Leaderboard
print("\n" + "="*100)
print(f"{'VIEWS':<25} | {'METHOD':<15} | {'TEST ACC (%)':<16} | {'PARAMS':<8} | {'TIME(avg)'} | {'RUNS'}")
print("-" * 100)

for res in results:
    views, method = res[0].split(" | ")
    # Format hiển thị kiểu: 95.36 ± 0.15
    acc_str = f"{res[1]:.2f} ± {res[2]:.2f}"
    
    print(f"{views:<25} | {method:<15} | {acc_str:<16} | {res[3]:>5.2f}M | {res[4]:>5.1f}m | {res[5]} folds")

print("="*100 + "\n")