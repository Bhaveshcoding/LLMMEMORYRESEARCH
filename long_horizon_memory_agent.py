# long_horizon_memory_agent.py
# FINAL VERSION WITH DEEPSEEK API INTEGRATION

import os
import json
import math
import time
import requests
import nltk
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
MEMORY_FILE = "memory.json"
FORGET_LOG_FILE = "forgetting_log.json"

# ==================== DEEPSEEK API SETUP ====================
# Your DeepSeek API Key
DEEPSEEK_API_KEY = "sk-0491edaf55cb43b49e606b4107d6da3c"  # Your key is here!
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


class SmartMemoryAgent:
    """Smart memory agent with DeepSeek AI for intelligent keyword extraction"""
    
    def __init__(self, decay_rate=0.01, forget_threshold=0.1, max_memories=50, use_api=True):
        """
        Args:
            decay_rate: How fast memories decay over time
            forget_threshold: Minimum relevance score to keep memory (0-1)
            max_memories: Maximum number of memories before pruning
            use_api: Whether to use DeepSeek API (recommended: True)
        """
        self.decay_rate = decay_rate
        self.forget_threshold = forget_threshold
        self.max_memories = max_memories
        self.use_api = use_api
        self.time_step = 0
        self.memory = []
        self.query_log = []
        self.forgetting_log = []
        self.load_memory()
        self.load_forgetting_log()
        
        # Test API connection
        if self.use_api:
            print("🔌 Testing DeepSeek API connection...")
            test_result = self.test_api_connection()
            if not test_result:
                print("⚠️  API connection failed. Switching to fallback mode.")
                self.use_api = False
            else:
                print("✅ DeepSeek API connected successfully!")
    
    def test_api_connection(self):
        """Test if DeepSeek API is working"""
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a test assistant."},
                    {"role": "user", "content": "Say 'API connected'"}
                ],
                "max_tokens": 10,
                "temperature": 0.1
            }
            
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def load_memory(self):
        """Load saved memories from file"""
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r') as f:
                    self.memory = json.load(f)
                
                # Ensure all memories have required fields
                for i, memory in enumerate(self.memory):
                    if 'id' not in memory:
                        memory['id'] = i
                    if 'original_text' not in memory:
                        memory['original_text'] = memory.get('text', '').lower()
                    if 'access_count' not in memory:
                        memory['access_count'] = 0
                    if 'timestamp' not in memory:
                        memory['timestamp'] = 0
                    if 'relevance_score' not in memory:
                        memory['relevance_score'] = 1.0
                    if 'meaningfulness_score' not in memory:
                        memory['meaningfulness_score'] = 1.0
                    if 'keywords' not in memory:
                        # Re-extract keywords for old memories
                        memory['keywords'] = self.extract_keywords(memory.get('text', ''))
                
                print(f"📂 Loaded {len(self.memory)} memories")
                
                # Set time step based on loaded memories
                if self.memory:
                    max_timestamp = max(mem.get('timestamp', 0) for mem in self.memory)
                    self.time_step = max_timestamp + 1
                    
            except Exception as e:
                print(f"❌ Error loading memory file: {e}")
                print("Starting with empty memory...")
                self.memory = []
        else:
            print("📭 No memory file found. Starting fresh.")
    
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
    
    def extract_keywords(self, text):
        """Extract keywords using DeepSeek API or fallback"""
        if not text:
            return []
        
        if self.use_api:
            return self.extract_keywords_with_deepseek(text)
        else:
            return self.extract_keywords_fallback(text)
    
    def extract_keywords_with_deepseek(self, text):
        """Extract keywords using DeepSeek AI API - SMART VERSION"""
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Smart prompt for keyword extraction
            prompt = f"""Analyze this text and extract ONLY the most important content keywords.
            Remove ALL filler words, greetings, emotions, and meaningless content.
            Focus on nouns, technical terms, and key concepts.
            Return ONLY a comma-separated list of 3-6 keywords, lowercase, no explanations.
            
            Text: "{text}"
            
            Keywords:"""
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are an expert at extracting meaningful keywords. Remove all filler words and meaningless content."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 100,
                "temperature": 0.2
            }
            
            print("🤖 Calling DeepSeek API for keyword extraction...")
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                keywords_text = result['choices'][0]['message']['content'].strip()
                
                # Clean and process the response
                keywords = []
                for k in keywords_text.split(','):
                    k = k.strip().lower()
                    k = k.replace('.', '').replace('!', '').replace('?', '')
                    if k and len(k) > 2 and k not in ['', 'keywords:', 'keyword:']:
                        keywords.append(k)
                
                # Filter out any remaining common meaningless words
                meaningless_words = {'hello', 'hi', 'hey', 'wow', 'amazing', 'awesome', 
                                   'beautiful', 'nice', 'good', 'great', 'cool', 'ok', 'okay'}
                keywords = [k for k in keywords if k not in meaningless_words]
                
                if keywords:
                    print(f"✅ DeepSeek extracted: {keywords}")
                    return keywords
                else:
                    print("⚠️  DeepSeek returned no keywords, using fallback")
                    return self.extract_keywords_fallback(text)
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")
                return self.extract_keywords_fallback(text)
                
        except requests.exceptions.Timeout:
            print("⏰ API timeout, using fallback")
            return self.extract_keywords_fallback(text)
        except Exception as e:
            print(f"⚠️  DeepSeek API error: {e}")
            return self.extract_keywords_fallback(text)
    
    def extract_keywords_fallback(self, text):
        """Fallback keyword extraction without API"""
        if not text:
            return []
            
        text = text.lower().strip()
        
        # Remove punctuation
        for char in ',.!?;:"\'()[]{}':
            text = text.replace(char, ' ')
        
        # Meaningless words to filter
        meaningless_words = {
            'hello', 'hi', 'hey', 'greetings', 'hey', 'hola',
            'beautiful', 'pretty', 'gorgeous', 'stunning', 'attractive', 'lovely',
            'nice', 'good', 'great', 'awesome', 'amazing', 'cool', 'excellent',
            'fantastic', 'wonderful', 'brilliant', 'splendid',
            'wow', 'oh', 'ah', 'oops', 'yay',
            'very', 'really', 'quite', 'rather', 'somewhat', 'pretty',
            'just', 'only', 'simply', 'merely', 'basically',
            'maybe', 'perhaps', 'possibly', 'probably', 'likely',
            'well', 'so', 'then', 'now', 'anyway', 'anyways', 'anyhow',
            'ok', 'okay', 'alright', 'right', 'sure',
            'please', 'thanks', 'thank', 'sorry', 'excuse',
            'um', 'uh', 'er', 'ah', 'hmm', 'hm', 'eh',
            'like', 'you know', 'i mean', 'sort of', 'kind of', 'type of',
            'actually', 'basically', 'literally', 'seriously', 'honestly',
            'thing', 'things', 'stuff', 'something', 'anything', 'everything'
        }
        
        # Domain-specific important words
        domain_words = {
            'ai', 'artificial', 'intelligence', 'machine', 'learning',
            'llm', 'bot', 'bots', 'agent', 'agents', 'assistant',
            'neural', 'network', 'networks', 'deep', 'learning',
            'python', 'programming', 'code', 'software', 'developer',
            'memory', 'memories', 'recall', 'remember', 'forgetting', 'forget',
            'decay', 'temporal', 'long-term', 'short-term',
            'sport', 'football', 'soccer', 'basketball', 'tennis', 'cricket',
            'study', 'studies', 'research', 'academic', 'university', 'college',
            'system', 'systems', 'build', 'building', 'create', 'creating',
            'develop', 'development', 'design', 'architecture'
        }
        
        words = text.split()
        keywords = []
        
        for word in words:
            word = word.strip()
            if len(word) < 2:
                continue
            
            # Skip meaningless words
            if word in meaningless_words:
                continue
            
            # Skip common stopwords
            if word in STOP_WORDS:
                continue
            
            # Keep domain words
            if word in domain_words:
                keywords.append(word)
            # Keep longer words (likely meaningful)
            elif len(word) >= 4:
                keywords.append(word)
        
        # Remove duplicates
        unique_keywords = list(set(keywords))
        
        # Sort: domain words first, then longer words
        def sort_key(word):
            if word in domain_words:
                return (0, -len(word))
            else:
                return (1, -len(word))
        
        unique_keywords.sort(key=sort_key)
        
        return unique_keywords[:6]
    
    def analyze_text_meaningfulness(self, text):
        """Analyze text and extract keywords with meaningfulness score"""
        print(f"\n🔍 Analyzing: '{text}'")
        
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
    
    def calculate_memory_relevance(self, memory):
        """Calculate current relevance score for a memory"""
        score = 1.0
        
        # Time decay
        age = self.time_step - memory.get('timestamp', 0)
        time_decay = math.exp(-self.decay_rate * age)
        score *= time_decay
        
        # Access frequency
        access_count = memory.get('access_count', 0)
        access_factor = min(access_count / 10, 1.0)
        score *= (0.3 + 0.7 * access_factor)
        
        # Recency of last access
        last_access = memory.get('last_accessed')
        if last_access is not None:
            last_access_age = self.time_step - last_access
            recency_factor = math.exp(-0.1 * last_access_age)
            score *= recency_factor
        
        return max(0.0, min(1.0, score))
    
    def store_memory(self, text):
        """Store a new memory with DeepSeek keyword analysis"""
        if not text:
            print("❌ Cannot store empty memory")
            return None
        
        print("\n" + "="*60)
        print("📝 STORE MEMORY WITH DEEPSEEK ANALYSIS")
        print("="*60)
        
        # Analyze text with DeepSeek
        keywords, meaningful_score = self.analyze_text_meaningfulness(text)
        
        # Check if we have meaningful content
        if not keywords:
            print(f"\n❌ REJECTED: No meaningful content found")
            print(f"   Text: '{text}'")
            return None
        
        # Check meaningfulness threshold
        if meaningful_score < 0.1:  # At least 10% meaningful
            print(f"\n❌ REJECTED: Low meaningfulness score ({meaningful_score:.2f} < 0.10)")
            choice = input("Store anyway? (yes/no): ").strip().lower()
            if choice != 'yes':
                return None
        
        # Check capacity
        if len(self.memory) >= self.max_memories:
            print("\n⚠️  Memory capacity reached. Applying forgetting...")
            self.apply_forgetting()
        
        # Create memory entry
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
    
    def apply_forgetting(self):
        """Apply forgetting mechanism to low-relevance memories"""
        print("\n🧠 APPLYING FORGETTING MECHANISM...")
        print("-"*40)
        
        memories_to_forget = []
        retained_memories = []
        
        # Calculate relevance for all memories
        for memory in self.memory:
            memory['relevance_score'] = self.calculate_memory_relevance(memory)
        
        # Sort by relevance (lowest first)
        self.memory.sort(key=lambda x: x['relevance_score'])
        
        for memory in self.memory:
            relevance = memory['relevance_score']
            
            # Criteria for forgetting
            should_forget = (
                relevance < self.forget_threshold or
                (len(self.memory) > self.max_memories and relevance < 0.5)
            )
            
            if should_forget:
                memories_to_forget.append(memory)
                forget_entry = {
                    'memory_id': memory['id'],
                    'text': memory['text'],
                    'relevance_score': relevance,
                    'forgotten_at': datetime.now().isoformat(),
                    'reason': 'low_relevance' if relevance < self.forget_threshold else 'capacity_limit'
                }
                self.forgetting_log.append(forget_entry)
                print(f"❌ Forgetting: '{memory['text'][:40]}...'")
            else:
                retained_memories.append(memory)
        
        self.memory = retained_memories
        self.save_memory()
        self.save_forgetting_log()
        
        print(f"\n📊 Forgetting Summary:")
        print(f"   Memories forgotten: {len(memories_to_forget)}")
        print(f"   Memories retained: {len(self.memory)}")
        
        return memories_to_forget
    
    def retrieve_memories(self, query):
        """Retrieve relevant memories for a query"""
        if not query:
            return []
        
        query_keywords = self.extract_keywords(query)
        
        scored_memories = []
        
        for memory in self.memory:
            memory_keywords = set(memory['keywords'])
            
            score = 0
            
            # Keyword overlap
            overlap = set(query_keywords) & memory_keywords
            if overlap:
                score += len(overlap) * 2.0
            
            # Relevance boost
            relevance = memory.get('relevance_score', 0.5)
            score *= (0.5 + 0.5 * relevance)
            
            if score > 0:
                # Update access stats
                memory['access_count'] = memory.get('access_count', 0) + 1
                memory['last_accessed'] = self.time_step
                scored_memories.append((score, memory))
        
        # Sort by score
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        
        return [memory for score, memory in scored_memories[:3]]
    
    def answer_query(self, query):
        """Generate answer for a query"""
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
        
        # Log query
        self.query_log.append({
            'query': query,
            'memories_found': len(memories),
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        })
        
        if not memories:
            return {
                'answer': f"I don't have information about '{query}'",
                'memories': [],
                'count': 0,
                'response_time': response_time
            }
        
        # Build answer
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
            'response_time': response_time
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
            print(f"📉 Average relevance: {sum(relevance_scores)/total:.3f}")
            
            # Show some memories
            print(f"\n📋 Sample memories:")
            for i, mem in enumerate(self.memory[:3], 1):
                print(f"  {i}. '{mem['text'][:50]}...'")
                print(f"     Keywords: {', '.join(mem['keywords'][:3])}")
        
        print("="*60)


def main():
    """Main program"""
    print("\n" + "="*60)
    print("🤖 SMART MEMORY AGENT WITH DEEPSEEK AI")
    print("="*60)
    
    # Create agent with DeepSeek API enabled
    agent = SmartMemoryAgent(
        decay_rate=0.02,
        forget_threshold=0.2,
        max_memories=20,
        use_api=True  # Enable DeepSeek API
    )
    
    # Store initial memories if empty
    if len(agent.memory) == 0:
        print("\n📝 SETTING UP INITIAL MEMORIES...")
        print("-"*40)
        
        initial_memories = [
            "My favorite sport is football and I play it every weekend.",
            "I study artificial intelligence and machine learning at university.",
            "Memory decay is important for long-term reasoning in AI systems.",
            "I want to build advanced AGI systems that can reason like humans."
        ]
        
        for text in initial_memories:
            print(f"\nStoring: '{text}'")
            agent.store_memory(text)
            time.sleep(1)  # Delay to avoid API rate limits
    else:
        print(f"\n✅ Found {len(agent.memory)} existing memories")
    
    # Main menu
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Test a query")
        print("2. Store new memory (with DeepSeek analysis)")
        print("3. Apply forgetting mechanism")
        print("4. Show memory health report")
        print("5. View all memories")
        print("6. Test DeepSeek keyword extraction")
        print("7. Exit")
        print("="*60)
        
        choice = input("Select option (1-7): ").strip()
        
        if choice == '1':
            print("\n" + "="*60)
            print("TEST QUERY")
            print("="*60)
            query = input("Enter query: ").strip()
            if query:
                result = agent.answer_query(query)
                agent.display_result(result, query)
        
        elif choice == '2':
            print("\n" + "="*60)
            print("STORE NEW MEMORY")
            print("="*60)
            text = input("Enter memory text: ").strip()
            if text:
                agent.store_memory(text)
        
        elif choice == '3':
            print("\n" + "="*60)
            print("APPLY FORGETTING")
            print("="*60)
            confirm = input("Apply forgetting mechanism? (yes/no): ").strip().lower()
            if confirm == 'yes':
                forgotten = agent.apply_forgetting()
                print(f"✅ Forgot {len(forgotten)} memories")
        
        elif choice == '4':
            agent.show_memory_health()
        
        elif choice == '5':
            print("\n" + "="*60)
            print("ALL MEMORIES")
            print("="*60)
            if agent.memory:
                for i, mem in enumerate(agent.memory, 1):
                    print(f"{i}. '{mem['text']}'")
                    print(f"   Keywords: {mem['keywords']}")
                    print()
            else:
                print("No memories stored.")
        
        elif choice == '6':
            print("\n" + "="*60)
            print("TEST DEEPSEEK KEYWORD EXTRACTION")
            print("="*60)
            text = input("Enter text to analyze: ").strip()
            if text:
                keywords, score = agent.analyze_text_meaningfulness(text)
                print(f"\n📊 Results:")
                print(f"   Keywords: {keywords}")
                print(f"   Score: {score:.2f}")
        
        elif choice == '7':
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
