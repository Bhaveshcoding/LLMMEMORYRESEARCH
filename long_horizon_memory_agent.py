import os
import json
import math
import time
import random
import re
import nltk
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import string
import httpx

load_dotenv()

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
MEMORY_FILE = "memory.json"
FORGET_LOG_FILE = "forgetting_log.json"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_API_URL,
    http_client=httpx.Client(verify=False)
)

class SmartMemoryAgent:
    """Smart memory agent with comprehensive DeepSeek AI integration"""
    
    def __init__(self, decay_rate=0.01, forget_threshold=0.1, max_memories=50, use_api=True):
        """
        Args:
            decay_rate: How fast memories decay over time
            forget_threshold: Minimum relevance score to keep memory (0-1)
            max_memories: Maximum number of memories before pruning
            use_api: Whether to use DeepSeek API
        """
        self.decay_rate = decay_rate
        self.forget_threshold = forget_threshold
        self.max_memories = max_memories
        self.use_api = use_api and bool(DEEPSEEK_API_KEY)
        self.time_step = 0
        self.memory = []
        self.query_log = []
        self.forgetting_log = []
        self.memory_categories = {}
        self.last_api_call = 0
        self.api_call_delay = 1.0
        
        self.load_memory()
        self.load_forgetting_log()

        if self.use_api:
            print("🔌 Testing DeepSeek API connection...")
            test_result = self.test_api_connection()
            if not test_result:
                print("⚠️  API connection failed. Switching to fallback mode.")
                self.use_api = False
            else:
                print("✅ DeepSeek API connected successfully!")
        else:
            print("⚠️  DeepSeek API disabled. Using fallback methods.")
    
    def test_api_connection(self):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a test assistant."},
                    {"role": "user", "content": "Say 'API connected'"}
                ],
                max_tokens=10,
                temperature=0.1,
                timeout=5
            )
            return bool(response.choices)
        except Exception:
            return False

    def call_deepseek_api(self, prompt, system_prompt="You are a helpful assistant.", max_tokens=500, temperature=0.3):
        """Generic function to call DeepSeek API using OpenAI client"""
        if not self.use_api:
            return ""

        now = time.time()
        delta = now - self.last_api_call
        if delta < self.api_call_delay:
            time.sleep(self.api_call_delay - delta)

        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=15
            )

            self.last_api_call = time.time()
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"⚠️  DeepSeek API error: {str(e)[:120]}")
            return ""
    
    def load_memory(self):
        """Start a fresh memory file for each session, timestamped"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_file = f"{MEMORY_FILE}.{timestamp}"

        self.memory = []
        self.time_step = 0
        self.memory_file = session_file

        try:
            with open(session_file, "w") as f:
                json.dump([], f)
            print(f"🧠 New memory session started: {session_file}")
        except Exception as e:
            print(f"❌ Failed to initialize memory file: {e}")
            self.memory = []
            self.memory_file = None    

    def load_forgetting_log(self):
        """Load forgetting history"""
        if os.path.exists(FORGET_LOG_FILE):
            try:
                with open(FORGET_LOG_FILE, 'r') as f:
                    self.forgetting_log = json.load(f)
            except:
                self.forgetting_log = []
    
    def save_memory(self):
        """Save memories to file"""
        with open(MEMORY_FILE, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def save_forgetting_log(self):
        """Save forgetting history"""
        with open(FORGET_LOG_FILE, 'w') as f:
            json.dump(self.forgetting_log, f, indent=2)
    
    def export_memories(self, filename="memory_export.json"):
        """Export memories to a file"""
        export_data = {
            'version': '1.0',
            'export_date': datetime.now().isoformat(),
            'memory_count': len(self.memory),
            'memories': self.memory
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✅ Memories exported to {filename}")
    
    def extract_keywords(self, text):
        """Extract keywords using DeepSeek API or fallback"""
        if not text:
            return []
        
        if self.use_api:
            return self.extract_keywords_with_deepseek(text)
        else:
            return self.extract_keywords_fallback(text)
    
    def extract_keywords_with_deepseek(self, text):
        """Extract high-signal keywords using DeepSeek"""
        try:
            prompt = (
                "Extract the core content keywords from the text.\n"
                "Rules:\n"
                "- 3 to 6 keywords only\n"
                "- nouns, technical terms, key concepts\n"
                "- lowercase\n"
                "- comma-separated\n"
                "- no filler, no emotions, no explanations\n\n"
                f"Text:\n{text}\n\n"
                "Keywords:"
            )

            response = self.call_deepseek_api(
                prompt=prompt,
                system_prompt=(
                    "You extract only meaningful content keywords. "
                    "You delete filler, greetings, opinions, and emotional language."
                ),
                max_tokens=60,
                temperature=0.1
            )

            if not response:
                return self.extract_keywords_fallback(text)

            raw = response.lower().strip()
            raw = raw.replace("\n", "").replace("keywords:", "")

            candidates = [
                k.strip(" .!?")
                for k in raw.split(",")
                if len(k.strip()) > 2
            ]

            blacklist = {
                "hello", "hi", "hey", "wow", "amazing", "awesome",
                "beautiful", "nice", "good", "great", "cool", "ok", "okay"
            }

            keywords = [k for k in candidates if k not in blacklist]

            if keywords:
                return keywords

            return self.extract_keywords_fallback(text)

        except Exception:
            return self.extract_keywords_fallback(text)
    
    def extract_keywords_fallback(self, text):
        if not text:
            return []

        text = text.lower()
        tokens = word_tokenize(text)

        stop = set(stopwords.words("english"))
        punct = set(string.punctuation)
        lemmatizer = WordNetLemmatizer()

        filtered = [
            lemmatizer.lemmatize(t)
            for t in tokens
            if t not in stop
            and t not in punct
            and len(t) > 2
            and t.isalpha()
        ]

        freq = Counter(filtered)
        return [w for w, _ in freq.most_common(6)]  
      
    def analyze_text_meaningfulness(self, text):
        """Analyze text and extract keywords with meaningfulness score"""
        print(f"\n🔍 Analyzing: '{text[:50]}...'" if len(text) > 50 else f"\n🔍 Analyzing: '{text}'")
        
        keywords = self.extract_keywords(text)
        
        # Calculate meaningfulness
        all_words = text.lower().split()
        cleaned_words = [w.strip('.,!?;:"\'()[]{}') for w in all_words if len(w.strip('.,!?;:"\'()[]{}')) > 1]
        
        if keywords and cleaned_words:
            meaningful_score = len(keywords) / len(cleaned_words)
        else:
            meaningful_score = 0.0
        
        print(f"Total words: {len(cleaned_words)}")
        print(f"Meaningful keywords: {len(keywords)}")
        
        if keywords:
            print(f"Keywords: {keywords}")
            print(f"Meaningfulness score: {meaningful_score:.2f}")
            
            if meaningful_score > 0.3:
                print("✅ High meaningfulness")
            elif meaningful_score > 0.1:
                print("⚠️  Moderate meaningfulness")
            else:
                print("❌ Low meaningfulness")
        else:
            print("❌ No meaningful content found")
        
        return keywords, meaningful_score
    
    def find_similar_memories(self, text, keywords, threshold=0.7):
        """Find similar existing memories"""
        similar = []
        for memory in self.memory:
            # Calculate similarity based on keyword overlap
            memory_keywords = set(memory['keywords'])
            new_keywords = set(keywords)
            
            if memory_keywords and new_keywords:
                similarity = len(memory_keywords & new_keywords) / len(memory_keywords | new_keywords)
                if similarity > threshold:
                    similar.append((similarity, memory))
        
        return similar
    
    def merge_memories(self, text, keywords, similar_memories):
        """Merge similar memories into one comprehensive memory"""
        # Get all similar memory texts
        memory_texts = [mem['text'] for _, mem in similar_memories]
        memory_texts.append(text)
        
        all_text = "\n".join([f"- {txt}" for txt in memory_texts])
        
        prompt = f"""Merge these similar memories into ONE comprehensive memory:
        
        {all_text}
        
        Create a single concise memory that includes all unique information.
        Remove redundancy but keep all important facts.
        Return ONLY the merged memory text."""
        
        merged_text = self.call_deepseek_api(prompt, "You are a memory consolidation expert.")
        
        if merged_text:
            # Extract new keywords from merged text
            new_keywords = self.extract_keywords(merged_text)
            
            # Calculate combined stats
            combined_access = sum(mem.get('access_count', 0) for _, mem in similar_memories) + 1
            max_relevance = max([mem.get('relevance_score', 0) for _, mem in similar_memories] + [1.0])
            
            # Create merged memory
            merged_memory = {
                'id': len(self.memory),
                'text': merged_text,
                'keywords': new_keywords,
                'timestamp': self.time_step,
                'access_count': combined_access,
                'original_text': merged_text.lower(),
                'creation_time': datetime.now().isoformat(),
                'last_accessed': None,
                'relevance_score': max_relevance,
                'meaningfulness_score': 1.0,
                'extraction_method': 'deepseek_merged',
                'merged_from': [mem['id'] for _, mem in similar_memories],
                'is_merged': True
            }
            
            # Remove original memories and add merged one
            memory_ids_to_remove = [mem['id'] for _, mem in similar_memories]
            self.memory = [mem for mem in self.memory if mem['id'] not in memory_ids_to_remove]
            self.memory.append(merged_memory)
            
            print(f"✅ Merged {len(similar_memories)} memories into 1")
            return merged_memory
        
        return None
    
    def store_memory(self, text):
        """Store a new memory with DeepSeek analysis and similarity check"""
        if not text:
            print("❌ Cannot store empty memory")
            return None
        
        print("\n" + "="*60)
        print("📝 STORE MEMORY WITH DEEPSEEK ANALYSIS")
        print("="*60)
        
        keywords, meaningful_score = self.analyze_text_meaningfulness(text)
        
        if not keywords:
            print(f"\n❌ REJECTED: No meaningful content found")
            print(f"   Text: '{text}'")
            return None
        
        if meaningful_score < 0.1:
            print(f"\n❌ REJECTED: Low meaningfulness score ({meaningful_score:.2f} < 0.10)")
            choice = input("Store anyway? (yes/no): ").strip().lower()
            if choice != 'yes':
                return None
        
        similar_memories = self.find_similar_memories(text, keywords, threshold=0.6)
        if similar_memories and self.use_api:
            print(f"\n⚠️  Found {len(similar_memories)} similar memories")
            print("Similar memories found:")
            for sim_score, mem in similar_memories:
                print(f"  - Similarity {sim_score:.2f}: '{mem['text'][:50]}...'")
            
            choice = input("\nMerge with existing memories? (yes/no): ").strip().lower()
            if choice == 'yes':
                merged = self.merge_memories(text, keywords, similar_memories)
                if merged:
                    self.save_memory()
                    return merged
        
        if len(self.memory) >= self.max_memories:
            print("\n⚠️  Memory capacity reached. Applying forgetting...")
            self.apply_forgetting()
        
        memory_entry = {
            'id': len(self.memory),
            'text': text,
            'keywords': keywords,
            'timestamp': self.time_step,
            'access_count': 0,
            'original_text': text.lower(),
            'creation_time': datetime.now().isoformat(),
            'last_accessed': None,
            'relevance_score': 1.0,
            'meaningfulness_score': meaningful_score,
            'extraction_method': 'deepseek' if self.use_api else 'fallback',
            'merged_from': []
        }
        
        self.memory.append(memory_entry)
        self.time_step += 1
        self.save_memory()
        
        print(f"\n✅ MEMORY STORED SUCCESSFULLY")
        print(f"   Text: '{text}'")
        print(f"   Keywords: {keywords}")
        print(f"   Meaningfulness: {meaningful_score:.2f}")
        print(f"   Memory count: {len(self.memory)}/{self.max_memories}")
        
        return memory_entry
    
    def summarize_memories_with_deepseek(self):
        if not self.use_api or len(self.memory) < 4:
            return

        groups = {}
        for m in self.memory:
            if m.get("is_summary") or m.get("is_merged"):
                continue
            key = m["keywords"][0] if m.get("keywords") else "other"
            groups.setdefault(key, []).append(m)

        created = 0

        for topic, mems in groups.items():
            if len(mems) < 3:
                continue

            prompt = (
                "Summarize the following related memories into one concise memory.\n"
                "Preserve all unique facts. No explanations.\n\n" +
                "\n".join(f"- {m['text']}" for m in mems) +
                "\n\nSummary:"
            )

            summary = self.call_deepseek_api(prompt, "Memory compression expert.", max_tokens=200, temperature=0.2)
            if not summary:
                continue

            new_mem = {
                "id": len(self.memory),
                "text": summary,
                "keywords": self.extract_keywords_with_deepseek(summary),
                "timestamp": self.time_step,
                "access_count": sum(m.get("access_count", 0) for m in mems),
                "original_text": summary.lower(),
                "creation_time": datetime.now().isoformat(),
                "relevance_score": max(m.get("relevance_score", 1.0) for m in mems),
                "meaningfulness_score": 1.0,
                "merged_from": [m["id"] for m in mems],
                "is_summary": True,
            }

            remove_ids = set(m["id"] for m in mems)
            self.memory = [m for m in self.memory if m["id"] not in remove_ids]
            self.memory.append(new_mem)
            created += 1

        if created:
            self.save_memory()

    def intelligent_forgetting_with_deepseek(self, candidate_memories):
        if not self.use_api or len(candidate_memories) < 2:
            return candidate_memories[:2]

        prompt = (
            "Select the 2–3 least valuable memories.\n"
            "Return only indices.\n\n" +
            "\n".join(f"{i+1}. {m['text']}" for i, m in enumerate(candidate_memories))
        )

        resp = self.call_deepseek_api(prompt, "Cognitive psychologist.", max_tokens=30, temperature=0.1)
        if not resp:
            return candidate_memories[:2]

        idx = [
            int(n) - 1 for n in resp.split(",")
            if n.strip().isdigit() and 0 <= int(n)-1 < len(candidate_memories)
        ]

        return [candidate_memories[i] for i in idx] if idx else candidate_memories[:2]
    
    def discover_memory_relationships_with_deepseek(self):
        if not self.use_api or len(self.memory) < 3:
            return ""

        sample = random.sample(self.memory, min(5, len(self.memory)))

        prompt = (
            "Analyze these memories for hidden themes, relationships, contradictions.\n\n" +
            "\n".join(f"{i+1}. {m['text']}" for i, m in enumerate(sample))
        )

        analysis = self.call_deepseek_api(prompt, "Pattern recognition expert.", max_tokens=400)
        if not analysis:
            return ""

        self.memory.append({
            "id": len(self.memory),
            "text": f"Pattern analysis: {analysis[:200]}",
            "keywords": ["patterns", "analysis"],
            "timestamp": self.time_step,
            "is_meta_memory": True,
            "analysis": analysis,
            "creation_time": datetime.now().isoformat()
        })

        self.save_memory()
        return analysis
    
    def personalized_retrieval_with_deepseek(self, query, user_context=""):
        base = self.retrieve_memories(query)
        if not self.use_api or not base:
            return base

        prompt = (
            f"Rank memories by relevance.\nQuery: {query}\nContext: {user_context}\n\n" +
            "\n".join(f"{i+1}. {m['text']}" for i, m in enumerate(base))
        )

        resp = self.call_deepseek_api(prompt, "Relevance ranking expert.", max_tokens=40, temperature=0.1)
        if not resp:
            return base

        order = [
            int(n) - 1 for n in resp.split(",")
            if n.strip().isdigit() and 0 <= int(n)-1 < len(base)
        ]

        seen = set(order)
        return [base[i] for i in order] + [m for i, m in enumerate(base) if i not in seen]
    
    def auto_categorize_memories_with_deepseek(self):
        if not self.use_api:
            return {}

        categories = {}
        valid = {"Personal", "Technical", "Facts", "Goals", "Trivia", "Other"}

        for m in self.memory:
            if "category" in m:
                continue

            prompt = (
                "Categorize into one: Personal, Technical, Facts, Goals, Trivia, Other.\n\n"
                f"Memory: {m['text']}"
            )

            cat = self.call_deepseek_api(prompt, max_tokens=10, temperature=0.0)
            m["category"] = cat if cat in valid else "Other"
            categories.setdefault(m["category"], []).append(m["id"])

        if categories:
            self.memory_categories = categories
            self.save_memory()

        return categories
    
    def assess_memory_quality_with_deepseek(self, memory_text):
        if not self.use_api:
            return None

        prompt = (
            "Score memory 1–10 on clarity, usefulness, uniqueness, emotional value, detail.\n"
            "Return JSON only.\n\n"
            f"Memory: {memory_text}"
        )

        resp = self.call_deepseek_api(prompt, "Memory evaluation expert.", max_tokens=200)
        if not resp:
            return None

        match = re.search(r"\{.*\}", resp, re.DOTALL)
        return json.loads(match.group()) if match else None
    
    def generate_memory_timeline_with_deepseek(self):
        if not self.use_api or len(self.memory) < 3:
            return ""

        mems = sorted(self.memory, key=lambda m: m.get("timestamp", 0))[-8:]

        prompt = (
            "Create a coherent narrative timeline.\n\n" +
            "\n".join(f"- {m['text']}" for m in mems)
        )

        return self.call_deepseek_api(prompt, "Narrative historian.", max_tokens=400)
    
    def calculate_memory_relevance(self, memory):
        """Calculate current relevance score for a memory"""
        score = 1.0
        age = self.time_step - memory.get('timestamp', 0)
        time_decay = math.exp(-self.decay_rate * age)
        score *= time_decay
        access_count = memory.get('access_count', 0)
        access_factor = min(access_count / 10, 1.0)
        score *= (0.3 + 0.7 * access_factor)
        last_access = memory.get('last_accessed')
        if last_access is not None:
            last_access_age = self.time_step - last_access
            recency_factor = math.exp(-0.1 * last_access_age)
            score *= recency_factor
        return max(0.0, min(1.0, score))
    
    def apply_forgetting(self):
        """Apply forgetting mechanism to low-relevance memories"""
        print("\n🧠 APPLYING FORGETTING MECHANISM...")
        print("-"*40)
        memories_to_forget = []
        retained_memories = []
        for memory in self.memory:
            memory['relevance_score'] = self.calculate_memory_relevance(memory)
        self.memory.sort(key=lambda x: x['relevance_score'])
        for memory in self.memory:
            relevance = memory['relevance_score']
            should_forget = (
                relevance < self.forget_threshold or
                (len(self.memory) > self.max_memories and relevance < 0.5)
            )
            if should_forget:
                memories_to_forget.append(memory)
        if len(memories_to_forget) > 3 and self.use_api:
            print(f"\n🤖 Using DeepSeek to select which memories to forget...")
            memories_to_forget = self.intelligent_forgetting_with_deepseek(memories_to_forget)
        memory_ids_to_forget = [mem['id'] for mem in memories_to_forget]
        retained_memories = [mem for mem in self.memory if mem['id'] not in memory_ids_to_forget]
        for memory in memories_to_forget:
            forget_entry = {
                'memory_id': memory['id'],
                'text': memory['text'],
                'relevance_score': memory['relevance_score'],
                'forgotten_at': datetime.now().isoformat(),
                'reason': 'low_relevance' if memory['relevance_score'] < self.forget_threshold else 'capacity_limit',
                'intelligent_forgetting': self.use_api
            }
            self.forgetting_log.append(forget_entry)
            print(f"❌ Forgetting: '{memory['text'][:40]}...'")
        self.memory = retained_memories
        self.save_memory()
        self.save_forgetting_log()
        print(f"\n📊 Forgetting Summary:")
        print(f"   Memories forgotten: {len(memories_to_forget)}")
        print(f"   Memories retained: {len(self.memory)}")
        print(f"   Method: {'Intelligent (DeepSeek)' if self.use_api else 'Basic'}")
        return memories_to_forget
    
    def retrieve_memories(self, query, use_personalized=False, user_context=""):
        """Retrieve relevant memories for a query"""
        if not query:
            return []
        
        query_keywords = self.extract_keywords(query)
        
        scored_memories = []
        
        for memory in self.memory:
            memory_keywords = set(memory['keywords'])
            score = 0
            overlap = set(query_keywords) & memory_keywords
            if overlap:
                score += len(overlap) * 2.0
            relevance = memory.get('relevance_score', 0.5)
            score *= (0.5 + 0.5 * relevance)
            if score > 0:
                memory['access_count'] = memory.get('access_count', 0) + 1
                memory['last_accessed'] = self.time_step
                scored_memories.append((score, memory))
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        standard_memories = [memory for score, memory in scored_memories[:5]]
        if use_personalized and self.use_api and user_context:
            return self.personalized_retrieval_with_deepseek(query, user_context)[:3]
        return standard_memories[:3]
    
    def answer_query(self, query, use_personalized=False, user_context=""):
        """Generate answer for a query"""
        if not query:
            return {
                'answer': "Please enter a query.",
                'memories': [],
                'count': 0,
                'response_time': 0
            }
        start_time = time.time()
        if use_personalized and self.use_api:
            memories = self.retrieve_memories(query, use_personalized=True, user_context=user_context)
            method = "Personalized (DeepSeek)"
        else:
            memories = self.retrieve_memories(query)
            method = "Standard"
        response_time = time.time() - start_time
        self.query_log.append({
            'query': query,
            'memories_found': len(memories),
            'response_time': response_time,
            'timestamp': datetime.now().isoformat(),
            'method': method
        })
        if not memories:
            return {
                'answer': f"I don't have information about '{query}'",
                'memories': [],
                'count': 0,
                'response_time': response_time,
                'method': method
            }
        memory_texts = [mem['text'] for mem in memories]
        if len(memory_texts) == 1:
            answer = f"I remember: {memory_texts[0]}"
        else:
            answer = "Based on my memories:\n"
            for i, text in enumerate(memory_texts, 1):
                answer += f"{i}. {text}\n"
        return {
            'answer': answer.strip(),
            'memories': memories,
            'count': len(memories),
            'response_time': response_time,
            'method': method
        }
    
    def display_result(self, result, query):
        """Display query result"""
        print("\n" + "="*60)
        print("QUERY RESULT")
        print("="*60)
        print(f"📋 Query: {query}")
        print(f"💭 Answer: {result['answer']}")
        print(f"📊 Memories found: {result['count']}")
        print(f"⏱️  Response time: {result['response_time']*1000:.1f} ms")
        print(f"🔧 Method: {result.get('method', 'Standard')}")
        
        if result['memories']:
            print("\n🔍 Retrieved memories:")
            for i, memory in enumerate(result['memories'], 1):
                print(f"  {i}. '{memory['text']}'")
                print(f"     Keywords: {', '.join(memory['keywords'][:3])}")
        print("="*60)
    
    def show_memory_health(self):
        """Show memory health statistics"""
        print("\n" + "="*60)
        print("🧠 MEMORY HEALTH REPORT")
        print("="*60)
        
        if not self.memory:
            print("No memories stored yet.")
            return
        
        total = len(self.memory)
        print(f"📊 Total memories: {total}/{self.max_memories}")
        print(f"📈 Capacity usage: {(total/self.max_memories)*100:.1f}%")
        
        if total > 0:
            relevance_scores = [m.get('relevance_score', 0) for m in self.memory]
            access_counts = [m.get('access_count', 0) for m in self.memory]
            print(f"📉 Average relevance: {sum(relevance_scores)/total:.3f}")
            print(f"📈 Average access count: {sum(access_counts)/total:.1f}")
            merged_count = sum(1 for m in self.memory if m.get('is_merged', False))
            summary_count = sum(1 for m in self.memory if m.get('is_summary', False))
            meta_count = sum(1 for m in self.memory if m.get('is_meta_memory', False))
            print(f"\n📋 Memory types:")
            print(f"   Regular: {total - merged_count - summary_count - meta_count}")
            print(f"   Merged: {merged_count}")
            print(f"   Summaries: {summary_count}")
            print(f"   Meta: {meta_count}")
            if self.memory_categories:
                print(f"\n🏷️  Categories:")
                for category, ids in self.memory_categories.items():
                    print(f"   {category}: {len(ids)}")
        
        print("="*60)
    
    def show_memory_categories(self):
        """Show memory categories"""
        if not self.memory_categories:
            print("\n⚠️  No categories assigned yet. Run auto-categorization first.")
            return
        
        print("\n" + "="*60)
        print("🏷️  MEMORY CATEGORIES")
        print("="*60)
        
        for category, memory_ids in self.memory_categories.items():
            print(f"\n{category.upper()}: {len(memory_ids)} memories")
            for mem_id in memory_ids[:3]:
                memory = next((m for m in self.memory if m['id'] == mem_id), None)
                if memory:
                    print(f"  - '{memory['text'][:50]}...'")
        
        print("="*60)


def main():
    """Main program"""
    print("\n" + "="*60)
    print("🤖 ENHANCED SMART MEMORY AGENT WITH DEEPSEEK AI")
    print("="*60)
    
    if not DEEPSEEK_API_KEY:
        print("⚠️  WARNING: No DEEPSEEK_API_KEY found in environment variables.")
        print("⚠️  Set it in .env file or environment: DEEPSEEK_API_KEY=your_key_here")
        print("⚠️  Some features will be disabled.")
        use_api = input("\nContinue without API? (yes/no): ").strip().lower() == 'yes'
        if not use_api:
            return
    else:
        use_api = True
    
    agent = SmartMemoryAgent(
        decay_rate=0.02,
        forget_threshold=0.2,
        max_memories=25,
        use_api=use_api
    )
    
    if len(agent.memory) == 0:
        print("\n📝 SETTING UP INITIAL MEMORIES...")
        print("-"*40)        
        initial_memories = [
            "My favorite sport is football and I play it every weekend.",
            "I study artificial intelligence and machine learning at university.",
            "Memory decay is important for long-term reasoning in AI systems.",
            "I want to build advanced AGI systems that can reason like humans.",
            "Python is my primary programming language for AI development.",
            "Neural networks require large amounts of training data to work well.",
            "Reinforcement learning is used for game playing AI agents.",
            "I enjoy watching football matches on weekends with friends."
        ]
        for text in initial_memories:
            print(f"\nStoring: '{text}'")
            agent.store_memory(text)
            if agent.use_api:
                time.sleep(1)
    else:
        print(f"\n✅ Found {len(agent.memory)} existing memories")
    while True:
        print("\n" + "="*60)
        print("MAIN MENU - ENHANCED SMART MEMORY AGENT")
        print("="*60)
        print("1. Test a query (Standard)")
        print("2. Test a query (Personalized with DeepSeek)")
        print("3. Store new memory (with DeepSeek analysis)")
        print("4. Apply forgetting mechanism")
        print("5. Show memory health report")
        print("6. View all memories")
        print("7. Test DeepSeek keyword extraction")
        print("8. 🤖 Summarize memories (DeepSeek)")
        print("9. 🔍 Discover patterns (DeepSeek)")
        print("10. 🏷️ Auto-categorize memories (DeepSeek)")
        print("11. 📊 Assess memory quality (DeepSeek)")
        print("12. 📅 Generate timeline (DeepSeek)")
        print("13. 🏷️ View memory categories")
        print("14. 💾 Export memories")
        print("15. Exit")
        print("="*60)
        
        choice = input("Select option (1-15): ").strip()
        
        if choice == '1':
            print("\n" + "="*60)
            print("TEST QUERY (STANDARD)")
            print("="*60)
            query = input("Enter query: ").strip()
            if query:
                result = agent.answer_query(query)
                agent.display_result(result, query)
        
        elif choice == '2':
            print("\n" + "="*60)
            print("TEST QUERY (PERSONALIZED)")
            print("="*60)
            query = input("Enter query: ").strip()
            if query:
                context = input("Enter context (optional): ").strip()
                result = agent.answer_query(query, use_personalized=True, user_context=context)
                agent.display_result(result, query)
        
        elif choice == '3':
            print("\n" + "="*60)
            print("STORE NEW MEMORY")
            print("="*60)
            text = input("Enter memory text: ").strip()
            if text:
                agent.store_memory(text)
        
        elif choice == '4':
            print("\n" + "="*60)
            print("APPLY FORGETTING")
            print("="*60)
            confirm = input("Apply forgetting mechanism? (yes/no): ").strip().lower()
            if confirm == 'yes':
                forgotten = agent.apply_forgetting()
                print(f"✅ Forgot {len(forgotten)} memories")
        
        elif choice == '5':
            agent.show_memory_health()
        
        elif choice == '6':
            print("\n" + "="*60)
            print("ALL MEMORIES")
            print("="*60)
            if agent.memory:
                for i, mem in enumerate(agent.memory, 1):
                    memory_type = ""
                    if mem.get('is_merged'):
                        memory_type = " [MERGED]"
                    elif mem.get('is_summary'):
                        memory_type = " [SUMMARY]"
                    elif mem.get('is_meta_memory'):
                        memory_type = " [META]"
                    
                    print(f"{i}. '{mem['text']}'{memory_type}")
                    print(f"   Keywords: {mem['keywords']}")
                    print(f"   Relevance: {mem.get('relevance_score', 0):.2f}")
                    print()
            else:
                print("No memories stored.")
        
        elif choice == '7':
            print("\n" + "="*60)
            print("TEST DEEPSEEK KEYWORD EXTRACTION")
            print("="*60)
            text = input("Enter text to analyze: ").strip()
            if text:
                keywords, score = agent.analyze_text_meaningfulness(text)
                print(f"\n📊 Results:")
                print(f"   Keywords: {keywords}")
                print(f"   Score: {score:.2f}")
        
        elif choice == '8':
            print("\n" + "="*60)
            print("SUMMARIZE MEMORIES WITH DEEPSEEK")
            print("="*60)
            confirm = input("This will merge similar memories. Continue? (yes/no): ").strip().lower()
            if confirm == 'yes':
                agent.summarize_memories_with_deepseek()
        
        elif choice == '9':
            print("\n" + "="*60)
            print("DISCOVER PATTERNS WITH DEEPSEEK")
            print("="*60)
            analysis = agent.discover_memory_relationships_with_deepseek()
            if analysis:
                print(f"\n🔍 Pattern Analysis:")
                print(analysis)
                print(f"\n✅ Analysis saved as meta-memory")
            else:
                print("❌ Failed to generate analysis")
        
        elif choice == '10':
            print("\n" + "="*60)
            print("AUTO-CATEGORIZE MEMORIES WITH DEEPSEEK")
            print("="*60)
            categories = agent.auto_categorize_memories_with_deepseek()
            if categories:
                print(f"\n✅ Categorized memories into {len(categories)} categories")
                for category, ids in categories.items():
                    print(f"   {category}: {len(ids)} memories")
            else:
                print("❌ No new memories to categorize")
        
        elif choice == '11':
            print("\n" + "="*60)
            print("ASSESS MEMORY QUALITY WITH DEEPSEEK")
            print("="*60)
            text = input("Enter memory text to assess: ").strip()
            if text:
                assessment = agent.assess_memory_quality_with_deepseek(text)
                if assessment:
                    print(f"\n📊 Quality Assessment:")
                    for key, value in assessment.items():
                        print(f"   {key}: {value}")
                else:
                    print("❌ Failed to assess memory quality")
        
        elif choice == '12':
            print("\n" + "="*60)
            print("GENERATE TIMELINE WITH DEEPSEEK")
            print("="*60)
            timeline = agent.generate_memory_timeline_with_deepseek()
            print(f"\n📅 Memory Timeline:")
            print(timeline)
        
        elif choice == '13':
            agent.show_memory_categories()
        
        elif choice == '14':
            print("\n" + "="*60)
            print("EXPORT MEMORIES")
            print("="*60)
            filename = input("Export filename (default: memory_export.json): ").strip()
            if not filename:
                filename = "memory_export.json"
            agent.export_memories(filename)
        
        elif choice == '15':
            print("\n" + "="*60)
            print("EXITING")
            print("="*60)
            agent.save_memory()
            agent.save_forgetting_log()
            print("✅ All data saved")
            break
        
        else:
            print("❌ Invalid choice")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
