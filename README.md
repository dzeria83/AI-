#!/usr/bin/env python3
"""
პრომეტე - უნივერსალური ქართული AI ასისტენტი
ავტომატურად არგებს მოდელს სისტემის რესურსების მიხედვით
"""

import os
import sys
import json
import torch
import psutil
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Union
import argparse
import warnings
warnings.filterwarnings("ignore")

# ==================== სისტემის აღმოჩენა ====================
class SystemDetector:
    @staticmethod
    def get_system_info() -> Dict:
        """ავტომატურად ამოიცნობს სისტემის რესურსებს"""
        info = {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "ram_total_gb": psutil.virtual_memory().total / (1024**3),
            "ram_available_gb": psutil.virtual_memory().available / (1024**3),
            "disk_free_gb": psutil.disk_usage('/').free / (1024**3) if os.name != 'nt' else psutil.disk_usage('C:').free / (1024**3),
            "is_android": 'android' in platform.system().lower() or 'ANDROID_ROOT' in os.environ,
            "is_mobile": platform.system() in ['Android', 'iOS', 'Darwin'] and 'Mobile' in platform.platform(),
            "has_gpu": torch.cuda.is_available() if torch else False,
            "cpu_cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True)
        }
        
        # GPU ინფო
        if info["has_gpu"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            info["gpu_name"] = "None"
            info["gpu_memory_gb"] = 0
            
        return info
    
    @staticmethod
    def recommend_model(system_info: Dict) -> str:
        """რეკომენდაცია მოდელის შესახებ რესურსების მიხედვით"""
        available_ram = system_info["ram_available_gb"]
        
        if available_ram < 1:
            return "error"  # ძალიან ცოტა RAM
        elif available_ram <= 2:
            return "micro"   # 1-2GB RAM
        elif available_ram <= 4:
            return "tiny"    # 2-4GB RAM
        elif available_ram <= 8:
            return "base"    # 4-8GB RAM
        elif available_ram <= 16:
            return "standard" # 8-16GB RAM
        elif available_ram <= 32:
            return "pro"     # 16-32GB RAM
        else:
            return "ultra"   # 32GB+ RAM

# ==================== მოდელების მენეჯერი ====================
class ModelManager:
    MODEL_CONFIGS = {
        "micro": {
            "name": "prometheus-micro",
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "quantization": "q4_0",
            "ram_required": 1.5,
            "storage_required": 0.5,
            "features": ["chat", "qa", "translation_basic", "summarization"],
            "languages": ["ka", "en"],
            "optimized_for": ["android", "low-end-pc", "raspberry-pi"]
        },
        "tiny": {
            "name": "prometheus-tiny",
            "base_model": "microsoft/phi-2",
            "quantization": "q4_K_M",
            "ram_required": 2.5,
            "storage_required": 1.2,
            "features": ["chat", "qa", "translation", "summarization", "sentiment", "entities"],
            "languages": ["ka", "en", "ru"],
            "optimized_for": ["android", "pc", "server"]
        },
        "base": {
            "name": "prometheus-base",
            "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "quantization": "q5_K_M",
            "ram_required": 4.0,
            "storage_required": 2.0,
            "features": ["chat", "qa", "translation", "summarization", "code", "reasoning"],
            "languages": ["ka", "en", "ru", "tr"],
            "optimized_for": ["pc", "server", "web"]
        },
        "standard": {
            "name": "prometheus-standard",
            "base_model": "google/gemma-2b-it",
            "quantization": "q6_K",
            "ram_required": 6.0,
            "storage_required": 3.0,
            "features": ["chat", "qa", "translation", "code", "reasoning", "creative"],
            "languages": ["ka", "en", "ru", "tr", "az"],
            "optimized_for": ["pc", "server", "web", "api"]
        },
        "pro": {
            "name": "prometheus-pro",
            "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
            "quantization": "q8_0",
            "ram_required": 12.0,
            "storage_required": 7.0,
            "features": ["all", "multimodal", "advanced_analysis", "long_context"],
            "languages": ["ka", "en", "ru", "tr", "az", "de", "fr"],
            "optimized_for": ["workstation", "server", "cloud"]
        },
        "ultra": {
            "name": "prometheus-ultra",
            "base_model": "deepseek-ai/deepseek-llm-7b-chat",
            "quantization": "none",
            "ram_required": 24.0,
            "storage_required": 14.0,
            "features": ["all", "multimodal", "advanced_reasoning", "research"],
            "languages": ["all_supported"],
            "optimized_for": ["server", "cloud", "enterprise"]
        }
    }
    
    @staticmethod
    def download_model(model_type: str, force_download: bool = False):
        """ავტომატური ჩამოტვირთვა"""
        config = ModelManager.MODEL_CONFIGS[model_type]
        model_path = Path(f"./models/{config['name']}")
        
        if model_path.exists() and not force_download:
            print(f"✅ მოდელი '{config['name']}' უკვე არსებობს")
            return str(model_path)
        
        print(f"📥 ჩამოტვირთვა: {config['name']}...")
        
        # აქ იქნება ჩამოტვირთვის ლოგიკა Hugging Face-დან
        # ტემპორარულად ვქმნით dummy მოდელს
        model_path.mkdir(parents=True, exist_ok=True)
        
        # შევქმნათ კონფიგურაციის ფაილი
        config_file = model_path / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        # შევქმნათ dummy მოდელი
        dummy_model = model_path / "model.bin"
        dummy_model.write_bytes(b'dummy_model_data')
        
        print(f"✅ მოდელი '{config['name']}' მზადაა")
        return str(model_path)

# ==================== AI ძრავა ====================
class PrometheusEngine:
    def __init__(self, model_type: str = "auto"):
        self.system_info = SystemDetector.get_system_info()
        
        if model_type == "auto":
            self.model_type = SystemDetector.recommend_model(self.system_info)
        else:
            self.model_type = model_type
            
        if self.model_type == "error":
            raise SystemError("❌ არასაკმარისი RAM! საჭიროა მინიმუმ 1GB.")
            
        self.config = ModelManager.MODEL_CONFIGS[self.model_type]
        self.model_path = ModelManager.download_model(self.model_type)
        self.model = None
        self.tokenizer = None
        
        print(f"\n{'='*50}")
        print(f"🤖 პრომეტე AI - {self.config['name']}")
        print(f"📊 სისტემა: {self.system_info['os']} | RAM: {self.system_info['ram_available_gb']:.1f}GB")
        print(f"🎯 მოდელი: {self.model_type} | ენები: {', '.join(self.config['languages'])}")
        print(f"⚡ ფუნქციები: {', '.join(self.config['features'][:5])}")
        print(f"{'='*50}\n")
        
        self._load_model()
    
    def _load_model(self):
        """მოდელის ჩატვირთვა"""
        print(f"🔄 მოდელის ჩატვირთვა...")
        
        try:
            # აქ იქნება რეალური მოდელის ჩატვირთვა
            # ტემპორარულად ვიყენებთ მარტივ ლოგიკას
            self.model = {"type": "dummy", "config": self.config}
            self.tokenizer = {"type": "dummy"}
            
            # კვანტიზაციის არჩევა
            if self.config["quantization"] != "none":
                print(f"🔧 კვანტიზაცია: {self.config['quantization']}")
            
            print(f"✅ მოდელი წარმატებით ჩაიტვირთა!")
            
        except Exception as e:
            print(f"❌ შეცდომა: {e}")
            self._fallback_to_lightweight()
    
    def _fallback_to_lightweight(self):
        """საჭიროების შემთხვევაში უფრო მსუბუქ მოდელზე გადასვლა"""
        print("🔄 უფრო მსუბუქ მოდელზე გადასვლა...")
        model_types = ["ultra", "pro", "standard", "base", "tiny", "micro"]
        
        for mt in model_types:
            if mt == self.model_type:
                continue
                
            req_ram = ModelManager.MODEL_CONFIGS[mt]["ram_required"]
            if self.system_info["ram_available_gb"] >= req_ram:
                self.model_type = mt
                self.config = ModelManager.MODEL_CONFIGS[mt]
                self.model_path = ModelManager.download_model(mt)
                print(f"✅ გადავედით: {self.config['name']}")
                break
    
    def process(self, prompt: str, language: str = "auto") -> str:
        """ტექსტის დამუშავება"""
        if language == "auto":
            # ენის ავტო-გამოცნობა
            if any(char in prompt for char in "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ"):
                language = "ka"
            else:
                language = "en"
        
        # სიმულაციური პასუხი
        responses = {
            "ka": [
                f"გამარჯობა! მე ვარ პრომეტე ({self.config['name']}).\n\nშენი შეკითხვა: '{prompt}'\n\nმე შემიძლია დაგეხმარო: {', '.join(self.config['features'])}.",
                f"პრომეტე აქ არის! რეჟიმი: {self.model_type}\n\nშეკითხვა: {prompt}\n\nპასუხი: ეს არის დემო პასუხი {self.config['name']} მოდელიდან."
            ],
            "en": [
                f"Hello! I'm Prometheus ({self.config['name']}).\n\nYour question: '{prompt}'\n\nI can help with: {', '.join(self.config['features'])}.",
                f"Prometheus here! Mode: {self.model_type}\n\nQuestion: {prompt}\n\nAnswer: This is a demo response from {self.config['name']} model."
            ]
        }
        
        import random
        return random.choice(responses.get(language, responses["en"]))
    
    def batch_process(self, prompts: List[str]) -> List[str]:
        """რამდენიმე შეკითხვის ერთდროულად დამუშავება"""
        return [self.process(p) for p in prompts]
    
    def get_capabilities(self) -> Dict:
        """მოდელის შესაძლებლობები"""
        return {
            "model": self.config["name"],
            "type": self.model_type,
            "features": self.config["features"],
            "languages": self.config["languages"],
            "ram_usage": f"{self.config['ram_required']}GB",
            "storage": f"{self.config['storage_required']}GB",
            "optimized_for": self.config["optimized_for"]
        }

# ==================== ინტერფეისები ====================
class InterfaceManager:
    @staticmethod
    def cli_interface(engine: PrometheusEngine):
        """კონსოლის ინტერფეისი"""
        print("\n🎮 CLI რეჟიმი (გასასვლელად: 'გამოსვლა' ან 'exit')")
        print("="*50)
        
        while True:
            try:
                user_input = input("\n🧑 > ").strip()
                
                if user_input.lower() in ['გამოსვლა', 'exit', 'quit', 'გამორთვა']:
                    print("👋 ნახვამდის!")
                    break
                elif user_input.lower() in ['ინფო', 'info', 'capabilities']:
                    caps = engine.get_capabilities()
                    print(f"\n📋 მოდელის ინფორმაცია:")
                    for key, value in caps.items():
                        print(f"  {key}: {value}")
                elif user_input.lower() in ['სისტემა', 'system', 'status']:
                    info = engine.system_info
                    print(f"\n🖥️ სისტემური სტატუსი:")
                    for key, value in info.items():
                        if 'gb' in key.lower():
                            print(f"  {key}: {value:.1f}GB")
                        else:
                            print(f"  {key}: {value}")
                elif user_input:
                    response = engine.process(user_input)
                    print(f"\n🤖 პრომეტე > {response}")
                    
            except KeyboardInterrupt:
                print("\n\n👋 დროებით!")
                break
            except Exception as e:
                print(f"\n❌ შეცდომა: {e}")
    
    @staticmethod
    def web_interface(engine: PrometheusEngine, port: int = 8080):
        """ვებ ინტერფეისის გაშვება"""
        print(f"🌐 ვებ ინტერფეისი: http://localhost:{port}")
        print("ℹ️ დასასრულებლად: Ctrl+C")
        
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import threading
            
            class WebHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/':
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html; charset=utf-8')
                        self.end_headers()
                        
                        html = f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <meta charset="UTF-8">
                            <title>პრომეტე AI</title>
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                                .container {{ max-width: 800px; margin: auto; }}
                                .prompt-box {{ width: 100%; height: 100px; }}
                                .response {{ background: #f5f5f5; padding: 20px; }}
                            </style>
                        </head>
                        <body>
                            <div class="container">
                                <h1>🤖 პრომეტე AI - {engine.config['name']}</h1>
                                <form method="POST">
                                    <textarea name="prompt" class="prompt-box" 
                                              placeholder="შეიყვანეთ თქვენი შეკითხვა..."></textarea><br>
                                    <button type="submit">გაგზავნა</button>
                                </form>
                                <div class="response" id="response">
                                    {engine.process("მოგესალმებით!")}
                                </div>
                            </div>
                        </body>
                        </html>
                        """
                        self.wfile.write(html.encode('utf-8'))
                
                def do_POST(self):
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length).decode('utf-8')
                    
                    # მარტივი პოსტ დატის დამუშავება
                    import urllib.parse
                    data = urllib.parse.parse_qs(post_data)
                    prompt = data.get('prompt', [''])[0]
                    
                    response = engine.process(prompt)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.end_headers()
                    
                    html = f"""
                    <!DOCTYPE html>
                    <html>
                    <body>
                        <div class="response">{response}</div>
                        <script>window.history.back();</script>
                    </body>
                    </html>
                    """
                    self.wfile.write(html.encode('utf-8'))
                
                def log_message(self, format, *args):
                    pass  # ლოგინგის გამორთვა
            
            server = HTTPServer(('localhost', port), WebHandler)
            print(f"✅ სერვერი გაშვებულია პორტზე {port}")
            server.serve_forever()
            
        except ImportError:
            print("❌ http.server არ არის ხელმისაწვდომი")
        except Exception as e:
            print(f"❌ შეცდომა: {e}")

# ==================== მთავარი ფუნქცია ====================
def main():
    parser = argparse.ArgumentParser(
        description="პრომეტე - უნივერსალური ქართული AI ასისტენტი",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
მაგალითები:
  %(prog)s                          # ავტომატური რეჟიმი
  %(prog)s --cli                    # CLI ინტერფეისი
  %(prog)s --web                    # ვებ ინტერფეისი
  %(prog)s --model tiny             # კონკრეტული მოდელი
  %(prog)s --prompt "გამარჯობა"     # ერთი შეკითხვა
  %(prog)s --batch ფაილი.txt        # ფაილიდან წაკითხვა
        """
    )
    
    parser.add_argument('--cli', action='store_true', help='CLI ინტერფეისი')
    parser.add_argument('--web', action='store_true', help='ვებ ინტერფეისი')
    parser.add_argument('--port', type=int, default=8080, help='ვებ პორტი')
    parser.add_argument('--model', choices=['micro', 'tiny', 'base', 'standard', 'pro', 'ultra', 'auto'], 
                       default='auto', help='მოდელის ტიპი')
    parser.add_argument('--prompt', type=str, help='პირდაპირი შეკითხვა')
    parser.add_argument('--batch', type=str, help='ფაილიდან წაკითხვა')
    parser.add_argument('--info', action='store_true', help='სისტემური ინფორმაცია')
    parser.add_argument('--download', action='store_true', help='მოდელების ჩამოტვირთვა')
    
    args = parser.parse_args()
    
    try:
        # სისტემური ინფორმაცია
        if args.info:
            detector = SystemDetector()
            info = detector.get_system_info()
            print(json.dumps(info, indent=2, ensure_ascii=False))
            return
        
        # მოდელების ჩამოტვირთვა
        if args.download:
            print("📥 ყველა მოდელის ჩამოტვირთვა...")
            for model_type in ['micro', 'tiny', 'base', 'standard', 'pro', 'ultra']:
                ModelManager.download_model(model_type)
            return
        
        # AI ძრავის შექმნა
        print("🔍 სისტემის ანალიზი...")
        engine = PrometheusEngine(model_type=args.model)
        
        # პირდაპირი შეკითხვა
        if args.prompt:
            response = engine.process(args.prompt)
            print(f"\n🤖 პასუხი:\n{response}\n")
            return
        
        # ფაილიდან წაკითხვა
        if args.batch:
            try:
                with open(args.batch, 'r', encoding='utf-8') as f:
                    prompts = [line.strip() for line in f if line.strip()]
                
                print(f"📖 ფაილიდან წაკითხვა: {len(prompts)} შეკითხვა")
                responses = engine.batch_process(prompts)
                
                for i, (prompt, response) in enumerate(zip(prompts, responses), 1):
                    print(f"\n{i}. ❓ {prompt}")
                    print(f"   🤖 {response}")
                    
            except FileNotFoundError:
                print(f"❌ ფაილი '{args.batch}' არ მოიძებნა")
            return
        
        # ვებ ინტერფეისი
        if args.web:
            InterfaceManager.web_interface(engine, args.port)
            return
        
        # CLI ინტერფეისი (სტანდარტული)
        InterfaceManager.cli_interface(engine)
        
    except SystemError as e:
        print(f"❌ {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 დროებით!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ კრიტიკული შეცდომა: {e}")
        sys.exit(1)

# ==================== გაშვება ====================
if __name__ == "__main__":
    # ვამოწმებთ დამოკიდებულებებს
    required_packages = ['psutil', 'torch']
    
    print("🔧 პრომეტე AI - ინიციალიზაცია...")
    
    # ვამოწმებთ Python ვერსიას
    if sys.version_info < (3, 8):
        print("❌ საჭიროა Python 3.8 ან უფრო მაღალი")
        sys.exit(1)
    
    main()
