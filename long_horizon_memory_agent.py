import os
import json
import math
import time
import random
import re
import nltk
from datetime import datetime
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from collections import Counter
import string
from nltk.sentiment import SentimentIntensityAnalyzer

load_dotenv()

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("vader_lexicon", quiet=True)


STOP_WORDS = set(stopwords.words('english'))
MEMORY_FILE = "memory.json"
FORGET_LOG_FILE = "forgetting_log.json"

class SmartMemoryAgent:
    """Smart memory agent with comprehensive DeepSeek AI integration"""
    
    def __init__(self, decay_rate=0.01, forget_threshold=0.1, max_memories=50):
        """
        Args:
            decay_rate: How fast memories decay over time
            forget_threshold: Minimum relevance score to keep memory (0-1)
            max_memories: Maximum number of memories before pruning
        """
        self.decay_rate = decay_rate
        self.forget_threshold = forget_threshold
        self.max_memories = max_memories
        self.time_step = 0
        self.memory = []
        self.query_log = []
        self.forgetting_log = []
        self.memory_categories = {}
        
        self.load_memory()
        self.load_forgetting_log()
    
    def load_memory(self):
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
        if os.path.exists(FORGET_LOG_FILE):
            try:
                with open(FORGET_LOG_FILE, 'r') as f:
                    self.forgetting_log = json.load(f)
            except:
                self.forgetting_log = []
    
    def save_memory(self):
        with open(MEMORY_FILE, 'w') as f:
            json.dump(self.memory, f, indent=2)
    
    def save_forgetting_log(self):
        with open(FORGET_LOG_FILE, 'w') as f:
            json.dump(self.forgetting_log, f, indent=2)
    
    def export_memories(self, filename="memory_export.json"):
        export_data = {
            'version': '1.0',
            'export_date': datetime.now().isoformat(),
            'memory_count': len(self.memory),
            'memories': self.memory
        } 
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"✅ Memories exported to {filename}")
    
    @staticmethod
    def extract_keywords(text, min_keywords=3, max_keywords=6):
        if not text:
            return []
        lemmatizer = WordNetLemmatizer()
        sia = SentimentIntensityAnalyzer()
        stop_words = set(stopwords.words("english")) - {"not", "no", "never"}
        punctuation = set(string.punctuation)
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        candidates = []
        for word, tag in tagged:
            if word in stop_words or word in punctuation or len(word) < 3 or not word.isalpha():
                continue
            if tag not in {"NN", "NNS", "NNP", "NNPS"}:
                continue
            if abs(sia.polarity_scores(word)["compound"]) > 0.3:
                continue
            candidates.append(lemmatizer.lemmatize(word))
        if not candidates:
            return []
        counts = Counter(candidates)
        ranked = [w for w, _ in counts.most_common()]
        return ranked[:max_keywords] if len(ranked) >= min_keywords else ranked
          
    def analyze_text_meaningfulness(self, text):
        print(f"\n🔍 Analyzing: '{text[:50]}...'" if len(text) > 50 else f"\n🔍 Analyzing: '{text}'")
        keywords = self.extract_keywords(text)
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
        similar = []
        for memory in self.memory:
            memory_keywords = set(memory['keywords'])
            new_keywords = set(keywords)
            if memory_keywords and new_keywords:
                similarity = len(memory_keywords & new_keywords) / len(memory_keywords | new_keywords)
                if similarity > threshold:
                    similar.append((similarity, memory))
        
        return similar

    def merge_memories(self, text, keywords, similar_memories):
        memory_texts = [mem['text'] for _, mem in similar_memories]
        memory_texts.append(text)
        merged_text = self.merge_with_nltk(memory_texts)
        if merged_text:
            new_keywords = self.extract_keywords(merged_text)
            combined_access = sum(mem.get('access_count', 0) for _, mem in similar_memories) + 1
            max_relevance = max([mem.get('relevance_score', 0) for _, mem in similar_memories] + [1.0])
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
                'extraction_method': 'fallback',
                'merged_from': [mem['id'] for _, mem in similar_memories],
                'is_merged': True
            }
            memory_ids_to_remove = [mem['id'] for _, mem in similar_memories]
            self.memory = [mem for mem in self.memory if mem['id'] not in memory_ids_to_remove]
            self.memory.append(merged_memory)
            print(f"✅ Merged {len(similar_memories)} memories into 1")
            return merged_memory
        return None

    def merge_with_nltk(self, memory_texts):
        all_sentences = []
        for text in memory_texts:
            sentences = nltk.sent_tokenize(text)
            all_sentences.extend(sentences)
        seen = set()
        unique_sentences = []
        for sentence in all_sentences:
            clean_sent = re.sub(r'\s+', ' ', sentence.strip()).lower()
            if clean_sent and clean_sent not in seen:
                seen.add(clean_sent)
                unique_sentences.append(sentence)
        all_tokens = []
        for text in memory_texts:
            tokens = nltk.word_tokenize(text.lower())
            tokens = [token for token in tokens if token.isalnum()]
            all_tokens.extend(tokens)
        word_freq = Counter(all_tokens)
        merged_text = " ".join(unique_sentences)
        if len(unique_sentences) > 0:
            merged_text = self.create_coherent_paragraph(unique_sentences, word_freq)
        else:
            merged_text = " ".join(memory_texts[:2])
        return merged_text

    def create_coherent_paragraph(self, sentences, word_freq):
        if len(sentences) <= 3:
            return " ".join(sentences)
        scored_sentences = []
        for sentence in sentences:
            score = 0
            tokens = nltk.word_tokenize(sentence.lower())
            for token in tokens:
                if token in word_freq:
                    score += 1.0 / (word_freq[token] + 1)
            scored_sentences.append((score, sentence))
        scored_sentences.sort(reverse=True)
        top_sentences = [sent for _, sent in scored_sentences[:min(5, len(scored_sentences))]]
        return " ".join(top_sentences)
    
    def store_memory(self, text):
        if not text:
            print("❌ Cannot store empty memory")
            return None
        print("\n" + "="*60)
        print("📝 STORE MEMORY")
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
        if similar_memories:
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
            'extraction_method': 'fallback',
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

    def intelligent_forgetting_with_deepseek(self, candidate_memories):
        if len(candidate_memories) < 2:
            return candidate_memories[:2]
        prompt = (
            "Select the 2-3 least valuable memories.\n"
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
        if len(memories_to_forget) > 3:
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
    
    def answer_query(self, query):
        if not query:
            return {
                'answer': "Please enter a query.",
                'memories': [],
                'count': 0,
                'response_time': 0
            }
        start_time = time.time()
        memories = self.retrieve_memories(query)
        response_time = time.time() - start_time
        self.query_log.append({
            'query': query,
            'memories_found': len(memories),
            'response_time': response_time,
            'timestamp': datetime.now().isoformat(),
            'method': "Fallback"
        })
        if not memories:
            return {
                'answer': f"I don't have information about '{query}'",
                'memories': [],
                'count': 0,
                'response_time': response_time,
                'method': "Fallback"
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
            'method': "Fallback"
        }
    
    def show_memory_health(self):
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
    print("🤖 ENHANCED SMART MEMORY AGENT")
    print("="*60)
    
    agent = SmartMemoryAgent(
        decay_rate=0.02,
        forget_threshold=0.2,
        max_memories=25,
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
            "I enjoy watching football matches on weekends with friends.",
            "Ayaan is a genuinely good student with strong habits.",
            "Bhavesh is absolutely goated at what he does.",
            "Sara puts in sincere effort every day.",
            "Rohan is dependable and easy to trust.",
            "Maya learns quickly and adapts well.",
            "Arjun stays focused and committed.",
            "Neha communicates clearly and thoughtfully.",
            "Karan shows admirable discipline.",
            "Isha has a solid understanding of the basics.",
            "Rahul reliably gets things done.",
            "Priya pays close and careful attention.",
            "Dev is steadily improving with consistency.",
            "Ananya always follows through on her work.",
            "Vikram keeps things simple and effective.",
            "Sana is impressively consistent.",
            "Amit understands his limits and works within them wisely.",
            "Pooja asks insightful questions.",
            "Nikhil delivers strong results.",
            "Meera quietly does the work and does it well.",
            "Bhavesh is a good student.",
            "Bhavesh is goated.",
            "Sara works hard.",
            "Rohan is reliable.",
            "Maya learns fast.",
            "Arjun stays focused.",
            "Neha speaks clearly.",
            "Karan shows discipline.",
            "Isha understands the basics.",
            "Rahul finishes tasks.",
            "Priya pays attention.",
            "Dev improves steadily.",
            "Ananya follows through.",
            "Vikram keeps it simple.",
            "Sana is consistent.",
            "Amit knows his limits.",
            "Pooja asks sharp questions.",
            "Nikhil gets results.",
            "Meera does the work."
        ]
        for text in initial_memories:
            print(f"\nStoring: '{text}'")
            agent.store_memory(text)
    else:
        print(f"\n✅ Found {len(agent.memory)} existing memories")
    while True:
        print("\n" + "="*60)
        print("MAIN MENU - ENHANCED SMART MEMORY AGENT")
        print("="*60)
        print("1. Test a query (Standard)")
        print("3. Store new memory")
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
            print("TEST KEYWORD EXTRACTION")
            print("="*60)
            text = input("Enter text to analyze: ").strip()
            if text:
                keywords, score = agent.analyze_text_meaningfulness(text)
                print(f"\n📊 Results:")
                print(f"   Keywords: {keywords}")
                print(f"   Score: {score:.2f}")
        
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