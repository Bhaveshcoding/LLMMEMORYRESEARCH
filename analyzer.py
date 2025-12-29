import json
import os
import matplotlib.pyplot as plt

class Analysis:
    def __init__(self, forget_log_folder, memory_log_folder):
        self.memory_log = []
        self.forget_log = []
        self._load_logs(forget_log_folder, memory_log_folder)
        self.analyze()

    def _load_logs(self, forget_log_folder, memory_log_folder):
        for root, dirs, files in os.walk(forget_log_folder):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(root, file), 'r') as f:
                        self.forget_log.extend(json.load(f))
        for root, dirs, files in os.walk(memory_log_folder):
            for file in files:
                if file.endswith('.json'):
                    with open(os.path.join(root, file), 'r') as f:
                        self.memory_log.extend(json.load(f))
    
    def analyze(self):
        pass
        plt.show()