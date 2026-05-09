import os
import numpy as np
import tflite_runtime.interpreter as tflite
import soundfile as sf
import resampy
import csv

# dung de xem RAM su dung
import psutil 


MODEL_PATH = "yamnet.tflite"
CSV_PATH = "yamnet_class_map.csv"
TARGET_FOLDER = "sounds_dataset/talk"

print("🚀 Đang khởi động YAMNet...")

# 1. Load từ điển nhãn
class_names = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        class_names.append(row['display_name'])

# 2. Load Model TFLite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def analyze_audio(file_path):
    try:
        data, sr = sf.read(file_path)
        
        # Chuyển về Mono nếu là Stereo
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
            
        # Resample về 16kHz
        if sr != 16000:
            data = resampy.resample(data, sr, 16000)
            
        # Lấy 15600 mẫu đầu tiên (0.975 giây) để test
        input_data = np.array(data[:15600], dtype=np.float32)
        if len(input_data) < 15600:
            input_data = np.pad(input_data, (0, 15600 - len(input_data)))
            
        # Chạy AI
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        scores = interpreter.get_tensor(output_details[0]['index'])[0]
        top_3_indices = np.argsort(scores)[-3:][::-1]
        
        return [(class_names[i], scores[i]) for i in top_3_indices]
    except Exception as e:
        return f"Lỗi: {e}"

# 3. Quét thư mục
if not os.path.exists(TARGET_FOLDER):
    print(f"❌ Không tìm thấy thư mục: {TARGET_FOLDER}. Bạn đã để file âm thanh vào đây chưa?")
else:
    print(f"📁 Bắt đầu phân tích các file trong: {TARGET_FOLDER}\n")
    print("-" * 50)
    
    files = [f for f in os.listdir(TARGET_FOLDER) if f.endswith('.wav')]
    if not files:
        print("Trống! Không có file .wav nào.")
        
    for filename in files:
        print(f"🎧 Đang nghe: {filename}")
        results = analyze_audio(os.path.join(TARGET_FOLDER, filename))
        
        if isinstance(results, list):
            for name, score in results:
                # Đánh dấu 🔥 nếu AI nghe thấy tiếng người/xì xầm
                marker = "🔥" if name in ["Speech", "Babble", "Chatter", "Conversation"] else "  "
                print(f"   {marker} ├─ {name}: {score:.2f}")
        else:
            print(f"   └─ {results}")
        print("-" * 50)


def print_memory_usage():
    # Lấy ID của tiến trình hiện tại (chính là script này)
    process = psutil.Process(os.getpid())
    
    # Lấy lượng RAM thực tế đang chiếm dụng (Resident Set Size - RSS)
    mem_bytes = process.memory_info().rss
    mem_mb = mem_bytes / (1024 * 1024)
    
    print(f"--- [RAM Usage]: {mem_mb:.2f} MB ---")
# Sau khi load mô hình YAMNet (đây là lúc RAM tăng mạnh nhất)
# model = load_yamnet_model() 
print("Đã load model...")
print_memory_usage()
