# long_horizon_memory_agent.py
# FINAL VERSION WITH COMPLETE DEEPSEEK INTEGRATION

import os
import json
import math
import time
import random
import re
import requests
import nltk
from datetime import datetime
from dotenv import load_dotenv
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load environment variables
load_dotenv()

# Download required data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
MEMORY_FILE = "memory.json"
FORGET_LOG_FILE = "forgetting_log.json"

# ==================== DEEPSEEK API SETUP ====================
# Get API key from environment variable
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


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
        self.api_call_delay = 1.0  # seconds between calls
        
        self.load_memory()
        self.load_forgetting_log()
        
        # Test API connection if enabled
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
    
    def call_deepseek_api(self, prompt, system_prompt="You are a helpful assistant.", max_tokens=500, temperature=0.3):
        """Generic function to call DeepSeek API with rate limiting"""
        if not self.use_api:
            return ""
        
        # Rate limiting
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.api_call_delay:
            time.sleep(self.api_call_delay - time_since_last_call)
        
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=15)
            self.last_api_call = time.time()
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                print(f"❌ API Error {response.status_code}: {response.text[:100]}")
        except requests.exceptions.Timeout:
            print("⏰ API timeout")
        except Exception as e:
            print(f"⚠️  DeepSeek API error: {e}")
        
        return ""
    
    def load_memory(self):
        """Load saved memories from file with error recovery"""
        if os.path.exists(MEMORY_FILE):
            try:
                # Try to load
                with open(MEMORY_FILE, 'r') as f:
                    data = f.read()
                    if not data.strip():
                        print("⚠️  Memory file is empty")
                        self.memory = []
                        return
                    
                    self.memory = json.loads(data)
                
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
                    
            except json.JSONDecodeError:
                print("⚠️  Memory file corrupted. Creating backup and starting fresh...")
                # Create backup
                backup_name = f"{MEMORY_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.rename(MEMORY_FILE, backup_name)
                print(f"✅ Backup created: {backup_name}")
                self.memory = []
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
        """Extract keywords using DeepSeek AI API - SMART VERSION"""
        try:
            # Smart prompt for keyword extraction
            prompt = f"""Analyze this text and extract ONLY the most important content keywords.
            Remove ALL filler words, greetings, emotions, and meaningless content.
            Focus on nouns, technical terms, and key concepts.
            Return ONLY a comma-separated list of 3-6 keywords, lowercase, no explanations.
            
            Text: "{text}"
            
            Keywords:"""
            
            keywords_text = self.call_deepseek_api(prompt, "You are an expert at extracting meaningful keywords. Remove all filler words and meaningless content.")
            
            if not keywords_text:
                print("⚠️  DeepSeek returned no response, using fallback")
                return self.extract_keywords_fallback(text)
            
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
                
        except Exception as e:
            print(f"⚠️  DeepSeek keyword extraction error: {e}")
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
        
        # Check for similar memories
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
    
    def summarize_memories_with_deepseek(self):
        """Use DeepSeek to summarize and compress similar memories"""
        if not self.use_api:
            print("❌ DeepSeek API required for summarization")
            return
        
        if len(self.memory) < 4:
            print("❌ Need at least 4 memories to summarize")
            return
        
        # Group memories by topic (simple keyword-based grouping)
        topic_groups = {}
        for memory in self.memory:
            if memory.get('is_merged', False) or memory.get('is_summary', False):
                continue
                
            main_keyword = memory['keywords'][0] if memory['keywords'] else 'other'
            if main_keyword not in topic_groups:
                topic_groups[main_keyword] = []
            topic_groups[main_keyword].append(memory)
        
        summaries_created = 0
        
        for topic, memories in topic_groups.items():
            if len(memories) >= 3:
                print(f"\n🤖 Summarizing {len(memories)} memories about: {topic}")
                
                memory_texts = "\n".join([f"- {mem['text']}" for mem in memories])
                
                prompt = f"""You are a memory compression expert. I have multiple related memories that need to be summarized into a single concise memory.
                
                Related memories:
                {memory_texts}
                
                Create ONE comprehensive summary that captures all key information. 
                Keep it concise but preserve all unique facts.
                Return ONLY the summary text without explanations.
                
                Summary:"""
                
                summary = self.call_deepseek_api(prompt, "You are a memory summarization expert.")
                
                if summary:
                    # Extract new keywords
                    keywords = self.extract_keywords_with_deepseek(summary)
                    
                    # Create summarized memory
                    summarized_memory = {
                        'id': len(self.memory),
                        'text': summary,
                        'keywords': keywords,
                        'timestamp': self.time_step,
                        'access_count': sum(mem.get('access_count', 0) for mem in memories),
                        'original_text': summary.lower(),
                        'creation_time': datetime.now().isoformat(),
                        'relevance_score': max(mem.get('relevance_score', 0) for mem in memories),
                        'meaningfulness_score': 1.0,
                        'extraction_method': 'deepseek_summary',
                        'merged_from': [mem['id'] for mem in memories],
                        'is_summary': True
                    }
                    
                    # Remove original memories and add summary
                    memory_ids_to_remove = [mem['id'] for mem in memories]
                    self.memory = [mem for mem in self.memory if mem['id'] not in memory_ids_to_remove]
                    self.memory.append(summarized_memory)
                    
                    summaries_created += 1
                    print(f"✅ Summarized {len(memories)} memories into 1")
        
        if summaries_created > 0:
            self.save_memory()
            print(f"\n📊 Total summaries created: {summaries_created}")
        else:
            print("\nℹ️  No suitable memory groups found for summarization")
    
    def intelligent_forgetting_with_deepseek(self, candidate_memories):
        """Use DeepSeek to decide which memories to forget intelligently"""
        if not self.use_api or len(candidate_memories) < 2:
            return candidate_memories[:min(2, len(candidate_memories))]
        
        # Ask DeepSeek to evaluate which memories are least valuable
        memory_list = "\n".join([f"{i+1}. '{mem['text']}'" for i, mem in enumerate(candidate_memories)])
        
        prompt = f"""As a cognitive psychologist, evaluate these memories for forgetting priority.
        Consider: usefulness, uniqueness, emotional value, and practical importance.
        
        Memories to evaluate:
        {memory_list}
        
        Return ONLY the numbers of the 2-3 least valuable memories to forget, separated by commas.
        For example: "2,5" or "1,3,4"
        Nothing else."""
        
        try:
            response = self.call_deepseek_api(prompt, "You are a cognitive psychologist specializing in memory retention.")
            indices_to_forget = []
            
            for num in response.strip().split(','):
                num = num.strip()
                if num.isdigit():
                    idx = int(num) - 1
                    if 0 <= idx < len(candidate_memories):
                        indices_to_forget.append(idx)
            
            if indices_to_forget:
                # Return memories to forget
                return [candidate_memories[i] for i in indices_to_forget]
            else:
                return candidate_memories[:min(2, len(candidate_memories))]
                
        except:
            return candidate_memories[:min(2, len(candidate_memories))]
    
    def discover_memory_relationships_with_deepseek(self):
        """Use DeepSeek to find hidden connections between memories"""
        if not self.use_api:
            print("❌ DeepSeek API required for pattern discovery")
            return ""
        
        if len(self.memory) < 3:
            print("❌ Need at least 3 memories to discover patterns")
            return ""
        
        # Sample memories for analysis
        sample_size = min(5, len(self.memory))
        sample_memories = random.sample(self.memory, sample_size)
        
        memory_texts = "\n".join([f"{i+1}: '{mem['text']}'" for i, mem in enumerate(sample_memories)])
        
        prompt = f"""You are a master of pattern recognition. Analyze these memories and find hidden connections, themes, or insights.
        
        Memories:
        {memory_texts}
        
        Identify:
        1. Common themes or patterns
        2. Interesting relationships between different memories
        3. Potential insights or conclusions
        4. Any contradictions or gaps
        
        Return as a structured analysis with clear sections."""
        
        analysis = self.call_deepseek_api(prompt, "You are an expert pattern recognizer and analyst.")
        
        # Store this analysis as a special meta-memory
        if analysis:
            meta_memory = {
                'id': len(self.memory),
                'text': f"Pattern Analysis: {analysis[:200]}..." if len(analysis) > 200 else f"Pattern Analysis: {analysis}",
                'keywords': ['meta-analysis', 'patterns', 'insights', 'relationships'],
                'timestamp': self.time_step,
                'is_meta_memory': True,
                'analysis': analysis,
                'creation_time': datetime.now().isoformat()
            }
            
            self.memory.append(meta_memory)
            self.save_memory()
        
        return analysis
    
    def personalized_retrieval_with_deepseek(self, query, user_context=""):
        """Use DeepSeek to understand query context and personalize retrieval"""
        # First, get standard results
        standard_memories = self.retrieve_memories(query)
        
        if not standard_memories or not self.use_api:
            return standard_memories
        
        # Use DeepSeek to re-rank based on context
        memories_list = "\n".join([f"{i+1}. '{mem['text']}'" for i, mem in enumerate(standard_memories)])
        
        prompt = f"""You are helping retrieve the most relevant memory. 
        
        User Query: "{query}"
        User Context: "{user_context}"
        
        Available memories:
        {memories_list}
        
        Return ONLY the numbers (1-{len(standard_memories)}) in order of relevance to the query AND context.
        Most relevant first, separated by commas.
        Example: "3,1,2" """
        
        ranked_order = self.call_deepseek_api(prompt, "You are a relevance ranking expert.")
        
        # Parse and re-order
        try:
            order = [int(num.strip()) - 1 for num in ranked_order.split(',') if num.strip().isdigit()]
            ordered_memories = [standard_memories[i] for i in order if i < len(standard_memories)]
            
            # Add any memories not in the ranking
            all_indices = set(range(len(standard_memories)))
            ranked_indices = set(order)
            unranked_indices = all_indices - ranked_indices
            
            for idx in unranked_indices:
                ordered_memories.append(standard_memories[idx])
                
            return ordered_memories
        except:
            return standard_memories
    
    def auto_categorize_memories_with_deepseek(self):
        """Use DeepSeek to automatically categorize memories"""
        if not self.use_api:
            print("❌ DeepSeek API required for categorization")
            return {}
        
        categories = {}
        categorized_count = 0
        
        for memory in self.memory:
            if 'category' in memory:
                continue
            
            prompt = f"""Categorize this memory into ONE of these categories:
            - Personal: about the user's life, feelings, experiences
            - Technical: about skills, work, technology, learning
            - Facts: objective information, data, facts
            - Goals: aspirations, plans, objectives
            - Trivia: interesting but not essential information
            - Other: doesn't fit above categories
            
            Memory: "{memory['text']}"
            
            Return ONLY the category name, nothing else."""
            
            category = self.call_deepseek_api(prompt).strip()
            
            # Validate category
            valid_categories = ['Personal', 'Technical', 'Facts', 'Goals', 'Trivia', 'Other']
            if category not in valid_categories:
                category = 'Other'
            
            memory['category'] = category
            categorized_count += 1
            
            if category not in categories:
                categories[category] = []
            categories[category].append(memory['id'])
        
        # Store category index
        self.memory_categories = categories
        
        if categorized_count > 0:
            self.save_memory()
            print(f"✅ Categorized {categorized_count} memories")
        
        return categories
    
    def assess_memory_quality_with_deepseek(self, memory_text):
        """Use DeepSeek to assess memory quality on multiple dimensions"""
        if not self.use_api:
            print("❌ DeepSeek API required for quality assessment")
            return None
        
        prompt = f"""Assess this memory on these dimensions (1-10 scale):
        
        Memory: "{memory_text}"
        
        Dimensions:
        1. Clarity: How clear and unambiguous is the memory?
        2. Usefulness: How likely is this to be useful in future?
        3. Uniqueness: Is this information unique or easily available elsewhere?
        4. Emotional value: Does this have personal significance?
        5. Detail level: Is it too vague or appropriately detailed?
        
        Return as a JSON object with scores and a brief explanation for each.
        Example format: {{"clarity": 7, "clarity_reason": "Clear but could be more specific", ...}}"""
        
        try:
            assessment = self.call_deepseek_api(prompt, "You are a memory quality assessment expert.")
            # Parse JSON from response
            json_match = re.search(r'\{.*\}', assessment, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"⚠️  Error parsing quality assessment: {e}")
        
        return None
    
    def generate_memory_timeline_with_deepseek(self):
        """Create a narrative timeline from memories"""
        if not self.use_api:
            print("❌ DeepSeek API required for timeline generation")
            return "DeepSeek API required for timeline generation"
        
        if len(self.memory) < 3:
            return "Need at least 3 memories to create a timeline"
        
        # Get recent memories
        recent_memories = sorted(self.memory, key=lambda x: x.get('timestamp', 0), reverse=True)[:8]
        recent_memories.reverse()  # Oldest first
        
        timeline_text = "\n".join([
            f"Memory {i+1}: {mem['text']}" 
            for i, mem in enumerate(recent_memories)
        ])
        
        prompt = f"""Create a coherent narrative timeline from these memories:
        
        {timeline_text}
        
        Organize them into a meaningful timeline with themes and progression.
        Return as a markdown timeline with dates/themes."""
        
        return self.call_deepseek_api(prompt, "You are a historian creating narrative timelines.")
    
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
        
        # Use intelligent forgetting if we have many candidates
        if len(memories_to_forget) > 3 and self.use_api:
            print(f"\n🤖 Using DeepSeek to select which memories to forget...")
            memories_to_forget = self.intelligent_forgetting_with_deepseek(memories_to_forget)
        
        # Actually forget the selected memories
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
        standard_memories = [memory for score, memory in scored_memories[:5]]
        
        # Apply personalized retrieval if requested
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
        
        # Log query
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
            # Calculate statistics
            relevance_scores = [m.get('relevance_score', 0) for m in self.memory]
            access_counts = [m.get('access_count', 0) for m in self.memory]
            
            print(f"📉 Average relevance: {sum(relevance_scores)/total:.3f}")
            print(f"📈 Average access count: {sum(access_counts)/total:.1f}")
            
            # Count memory types
            merged_count = sum(1 for m in self.memory if m.get('is_merged', False))
            summary_count = sum(1 for m in self.memory if m.get('is_summary', False))
            meta_count = sum(1 for m in self.memory if m.get('is_meta_memory', False))
            
            print(f"\n📋 Memory types:")
            print(f"   Regular: {total - merged_count - summary_count - meta_count}")
            print(f"   Merged: {merged_count}")
            print(f"   Summaries: {summary_count}")
            print(f"   Meta: {meta_count}")
            
            # Show categories if available
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
            for mem_id in memory_ids[:3]:  # Show first 3
                memory = next((m for m in self.memory if m['id'] == mem_id), None)
                if memory:
                    print(f"  - '{memory['text'][:50]}...'")
        
        print("="*60)


def main():
    """Main program"""
    print("\n" + "="*60)
    print("🤖 ENHANCED SMART MEMORY AGENT WITH DEEPSEEK AI")
    print("="*60)
    
    # Check for API key
    if not DEEPSEEK_API_KEY:
        print("⚠️  WARNING: No DEEPSEEK_API_KEY found in environment variables.")
        print("⚠️  Set it in .env file or environment: DEEPSEEK_API_KEY=your_key_here")
        print("⚠️  Some features will be disabled.")
        use_api = input("\nContinue without API? (yes/no): ").strip().lower() == 'yes'
        if not use_api:
            return
    else:
        use_api = True
    
    # Create agent
    agent = SmartMemoryAgent(
        decay_rate=0.02,
        forget_threshold=0.2,
        max_memories=25,
        use_api=use_api
    )
    
    # Store initial memories if empty
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
                time.sleep(1)  # Delay to avoid API rate limits
    else:
        print(f"\n✅ Found {len(agent.memory)} existing memories")
    
    # Main menu
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
