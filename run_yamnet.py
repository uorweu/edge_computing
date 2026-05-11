import os
import numpy as np
import tflite_runtime.interpreter as tflite
import soundfile as sf
import resampy
import csv
import shutil
import sys

# --- 1. CẤU HÌNH ---
MODEL_PATH = "yamnet.tflite"
CSV_PATH = "yamnet_class_map.csv"
TARGET_FOLDER = "my_dataset/Telephone bell ringing" # Chỉnh đường dẫn ở đây
ERROR_FOLDER = "needs_review"
CONFIDENCE_THRESHOLD = 0.8 

# --- 2. BẢNG ÁNH XẠ ---
LABEL_MAP = {
    "Laughter": ["Laughter", "Chuckle, giggle", "Baby laughter", "Giggl"],
    "Chewing, mastication": ["Chewing, mastication", "Biting", "Eating", "Crunch", "Mouth sound"],
    "Keyboard typing": ["Keyboard typing", "Computer keyboard", "Clicking"],
    "Squeak": ["Squeak", "Friction", "Door", "Tools"],
    "Crowd": ["Crowd", "Chatter", "People babbling", "Speech"],
    "Clicking": ["Clicking", "Mechanical click", "Writing", "Tick"],
    "Telephone bell ringing": ["Telephone bell", "Ringtone", "Telephone", "Alarm clock", "Beep", "Busy signal", "Alarm"],
    "bookflip": ["Pages turning", "Paper", "Shatter"],
    "Footstep": ["Walk, footsteps"],
    "Talking": ["Speech" , "Child speech, kid speaking"]
}

# --- 3. KHỞI TẠO AI ---
class_names = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row['display_name'])

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def analyze_audio_safe(file_path):
    """Chiến thuật mới: Đọc từng mẩu nhỏ từ đĩa cứng để không bao giờ treo RAM"""
    try:
        all_scores = []
        with sf.SoundFile(file_path) as f:
            sr = f.samplerate
            # Quét từng khối 1 giây (sr mẫu mỗi lần)
            while f.tell() < f.frames:
                chunk = f.read(sr) # Đọc đúng 1 giây
                
                if len(chunk.shape) > 1: chunk = np.mean(chunk, axis=1)
                
                # Chỉ resample mẩu 1 giây này (Cực nhanh)
                if sr != 16000:
                    chunk = resampy.resample(chunk, sr, 16000)
                
                input_data = np.array(chunk[:15600], dtype=np.float32)
                if len(input_data) < 15600:
                    input_data = np.pad(input_data, (0, 15600 - len(input_data)))
                
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                all_scores.append(interpreter.get_tensor(output_details[0]['index'])[0])
                
                # Giới hạn tối đa quét 30 đoạn để tránh file quá dài (tùy chọn)
                if len(all_scores) >= 30: break 

        if not all_scores: return "Silence", 0
        avg_scores = np.mean(all_scores, axis=0)
        top_idx = np.argmax(avg_scores)
        return class_names[top_idx], avg_scores[top_idx]
    except Exception as e:
        return f"Lỗi: {str(e)[:10]}", 0

# --- 4. CHƯƠNG TRÌNH CHÍNH ---
print(f"🚀 Đang quét: {TARGET_FOLDER}", flush=True)

if not os.path.exists(TARGET_FOLDER):
    print(f"❌ Không thấy folder: {TARGET_FOLDER}")
    sys.exit()

# Tự động nhận diện folder lẻ hoặc tổng
wavs = [f for f in os.listdir(TARGET_FOLDER) if f.endswith('.wav')]
if wavs:
    folders_to_scan = [(os.path.basename(TARGET_FOLDER.strip('/')), TARGET_FOLDER)]
else:
    sub = sorted([d for d in os.listdir(TARGET_FOLDER) if os.path.isdir(os.path.join(TARGET_FOLDER, d))])
    folders_to_scan = [(d, os.path.join(TARGET_FOLDER, d)) for d in sub]

summary = []
for group_name, folder_path in folders_to_scan:
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.wav')])
    if not files: continue
    
    print(f"\n📂 Nhóm: [{group_name}]")
    print("-" * 65, flush=True)
    correct = 0

    for filename in files:
        path = os.path.join(folder_path, filename)
        pred, score = analyze_audio_safe(path)
        
        valid = LABEL_MAP.get(group_name, [group_name])
        is_match = any(v.lower().strip() in pred.lower().strip() for v in valid)

        if is_match and score >= CONFIDENCE_THRESHOLD:
            correct += 1
            print(f" ✅ {filename[:15]:<15} | Đúng ({pred} - {score:.2f})", flush=True)
        else:
            print(f" ❌ {filename[:15]:<15} | Sai (AI đoán: {pred} - {score:.2f})", flush=True)
            err_p = os.path.join(ERROR_FOLDER, group_name)
            os.makedirs(err_p, exist_ok=True)
            shutil.copy(path, os.path.join(err_p, filename))
    
    summary.append({'name': group_name, 'correct': correct, 'total': len(files)})

# --- 5. TỔNG KẾT ---
print("\n" + "="*75)
print(f"{'TÊN THƯ MỤC':<30} | {'ĐÚNG/TỔNG':<15} | {'ĐỘ CHÍNH XÁC'}")
for s in summary:
    acc = (s['correct'] / s['total']) * 100
    print(f"{s['name']:<30} | {s['correct']:>5}/{s['total']:<9} | {acc:>10.2f}%")
