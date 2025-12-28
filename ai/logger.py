from datetime import datetime
import json
import os


class FileHandler:
    def __init__(self, forget_prefix="forgetting_log", memory_prefix="memory"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs("memory", exist_ok=True)
        os.makedirs("forget", exist_ok=True)

        self.forget_file = os.path.join("forget", f"{forget_prefix}_{timestamp}.json")
        self.memory_file = os.path.join("memory", f"{memory_prefix}_{timestamp}.json")
        self.memory = []
        self.forgetting_log = []
        self.time_step = 0

    def load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.memory = json.load(f)
                print(f"✅ Loaded {len(self.memory)} memories from {self.memory_file}")
            except Exception as e:
                print(f"❌ Failed to load memory file: {e}")
                self.memory = []
        else:
            try:
                with open(self.memory_file, "w") as f:
                    json.dump([], f)
                print(f"🧠 New memory session started: {self.memory_file}")
            except Exception as e:
                print(f"❌ Failed to initialize memory file: {e}")
                self.memory = []

    def load_forgetting_log(self):
        if os.path.exists(self.forget_file):
            try:
                with open(self.forget_file, 'r') as f:
                    self.forgetting_log = json.load(f)
            except:
                self.forgetting_log = []

    def save_memory(self):
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to save memory: {e}")

    def save_forgetting_log(self):
        try:
            with open(self.forget_file, 'w') as f:
                json.dump(self.forgetting_log, f, indent=2)
        except Exception as e:
            print(f"❌ Failed to save forgetting log: {e}")

    def export_memories(self, filename="memory_export.json"):
        export_data = {
            'version': '1.0',
            'export_date': datetime.now().isoformat(),
            'memory_count': len(self.memory),
            'memories': self.memory
        }
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"✅ Memories exported to {filename}")
        except Exception as e:
            print(f"❌ Failed to export memories: {e}")

    def load_from_jsonb(self, data_bytes):
        try:
            data = json.loads(data_bytes.decode('utf-8'))
            self.memory = data['memory']
            self.time_step = data['time_step']
            return True
        except Exception as e:
            print(f"❌ Failed to load from JSONB: {e}")
            return False

    def save_to_jsonb(self):
        try:
            data = {
                'memory': self.memory,
                'time_step': self.time_step
            }
            return json.dumps(data).encode('utf-8')
        except Exception as e:
            print(f"❌ Failed to save to JSONB: {e}")
            return None