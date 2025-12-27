# long_horizon_memory_agent.py
# COMPLETE FIXED VERSION WITH ALL OPTIONS WORKING

import os
import json
import math
import time
import random
import nltk
from datetime import datetime, timedelta
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
MEMORY_FILE = "memory.json"
FORGET_LOG_FILE = "forgetting_log.json"


class AdvancedMemoryAgent:
    """Advanced memory agent with forgetting mechanism"""
    
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
        self.load_memory()
        self.load_forgetting_log()
    
    def load_memory(self):
        """Load saved memories with backward compatibility"""
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, 'r') as f:
                    self.memory = json.load(f)
                
                # Fix old memory format
                for memory in self.memory:
                    # Ensure all required keys exist
                    if 'original_text' not in memory:
                        memory['original_text'] = memory.get('text', '').lower()
                    if 'access_count' not in memory:
                        memory['access_count'] = 0
                    if 'keywords' not in memory:
                        memory['keywords'] = self.extract_keywords(memory.get('text', ''))
                    if 'timestamp' not in memory:
                        memory['timestamp'] = 0
                    if 'id' not in memory:
                        memory['id'] = len(self.memory)
                    if 'creation_time' not in memory:
                        memory['creation_time'] = datetime.now().isoformat()
                    if 'last_accessed' not in memory:
                        memory['last_accessed'] = None
                    if 'relevance_score' not in memory:
                        memory['relevance_score'] = 1.0  # Initial score
                
                print(f"📂 Loaded {len(self.memory)} memories")
                
                # Update time_step based on loaded memories
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
        """Extract keywords from text"""
        if not text:
            return []
            
        text = text.lower().strip()
        
        # Remove question marks and common question words
        text = text.replace('?', '')
        question_words = ['what', 'do', 'you', 'i', 'my', 'me', 'your', 'is', 'are', 'tell', 'about']
        for word in question_words:
            text = text.replace(word, '')
        
        # Split into words
        words = text.split()
        keywords = []
        
        for word in words:
            # Clean the word
            word = word.strip('.,!?;:"\'()[]{}')
            if len(word) < 2:
                continue
            
            # Keep content words (not stopwords)
            if word not in STOP_WORDS:
                keywords.append(word)
        
        # Add important words if they appear
        important_words = ['sport', 'football', 'study', 'artificial', 
                          'intelligence', 'ai', 'memory', 'decay', 'build',
                          'systems', 'favorite', 'like', 'love', 'enjoy',
                          'programming', 'python', 'learning', 'machine',
                          'neural', 'networks', 'data', 'training', 'forget',
                          'forgetting', 'recall', 'remember', 'retention']
        
        for word in important_words:
            if word in text and word not in keywords:
                keywords.append(word)
        
        return keywords[:8]
    
    def calculate_memory_relevance(self, memory):
        """Calculate current relevance score for a memory"""
        # Base score starts at 1.0
        score = 1.0
        
        # 1. Time decay (older memories are less relevant)
        age = self.time_step - memory.get('timestamp', 0)
        time_decay = math.exp(-self.decay_rate * age)
        score *= time_decay
        
        # 2. Access frequency (rarely accessed memories fade)
        access_count = memory.get('access_count', 0)
        access_factor = min(access_count / 10, 1.0)  # Normalize to 0-1
        score *= (0.3 + 0.7 * access_factor)  # 30% base + 70% based on access
        
        # 3. Recency of last access
        last_access = memory.get('last_accessed')
        if last_access is not None:
            last_access_age = self.time_step - last_access
            recency_factor = math.exp(-0.1 * last_access_age)
            score *= recency_factor
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
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
        
        # Forget memories below threshold OR if we have too many
        for memory in self.memory:
            relevance = memory['relevance_score']
            
            # Criteria for forgetting:
            # 1. Relevance below threshold
            # 2. OR we have too many memories and this one is low relevance
            should_forget = (
                relevance < self.forget_threshold or
                (len(self.memory) > self.max_memories and 
                 relevance < 0.5)  # Extra pruning when over capacity
            )
            
            if should_forget:
                memories_to_forget.append(memory)
                
                # Log the forgetting
                forget_entry = {
                    'memory_id': memory['id'],
                    'text': memory['text'],
                    'relevance_score': relevance,
                    'access_count': memory.get('access_count', 0),
                    'age': self.time_step - memory.get('timestamp', 0),
                    'forgotten_at': datetime.now().isoformat(),
                    'reason': 'low_relevance' if relevance < self.forget_threshold else 'capacity_limit'
                }
                self.forgetting_log.append(forget_entry)
                
                print(f"❌ Forgetting: '{memory['text'][:40]}...'")
                print(f"   Score: {relevance:.3f}, Age: {self.time_step - memory.get('timestamp', 0)} steps")
            else:
                retained_memories.append(memory)
        
        # Update memory list
        self.memory = retained_memories
        
        # Save logs
        self.save_memory()
        self.save_forgetting_log()
        
        print(f"\n📊 Forgetting Summary:")
        print(f"   Memories forgotten: {len(memories_to_forget)}")
        print(f"   Memories retained: {len(self.memory)}")
        print(f"   New memory count: {len(self.memory)}/{self.max_memories}")
        
        return memories_to_forget
    
    def consolidate_memories(self):
        """Merge similar memories to reduce redundancy - FIXED VERSION"""
        print("\n🔄 CONSOLIDATING SIMILAR MEMORIES...")
        print("-"*40)
        
        if len(self.memory) < 2:
            print("   Not enough memories to consolidate")
            return
        
        consolidated = []
        processed = set()
        
        for i, mem1 in enumerate(self.memory):
            if i in processed:
                continue
                
            similar_memories = [mem1]
            
            # Find similar memories
            for j, mem2 in enumerate(self.memory[i+1:], i+1):
                if j in processed:
                    continue
                
                # Calculate similarity (keyword overlap)
                keywords1 = set(mem1['keywords'])
                keywords2 = set(mem2['keywords'])
                overlap = len(keywords1 & keywords2) / max(len(keywords1 | keywords2), 1)
                
                if overlap > 0.5:  # 50% keyword overlap
                    similar_memories.append(mem2)
                    processed.add(j)
            
            if len(similar_memories) > 1:
                # Merge similar memories
                merged_text = self.merge_memories(similar_memories)
                merged_keywords = self.extract_keywords(merged_text)
                
                # Keep the most recent/accessed memory as base
                base_memory = max(similar_memories, 
                                 key=lambda x: (
                                     x.get('access_count', 0), 
                                     x.get('timestamp', 0)
                                 ))
                
                # Safely get last_accessed values, handling None
                last_accessed_values = []
                for m in similar_memories:
                    la = m.get('last_accessed')
                    if la is not None:
                        last_accessed_values.append(la)
                
                # Calculate max last_accessed (if any exist)
                max_last_accessed = max(last_accessed_values) if last_accessed_values else None
                
                consolidated_memory = {
                    'id': base_memory['id'],
                    'text': merged_text,
                    'keywords': merged_keywords,
                    'timestamp': base_memory['timestamp'],
                    'access_count': sum(m.get('access_count', 0) for m in similar_memories),
                    'original_text': merged_text.lower(),
                    'creation_time': base_memory.get('creation_time', datetime.now().isoformat()),
                    'last_accessed': max_last_accessed,
                    'relevance_score': max(m.get('relevance_score', 0) for m in similar_memories),
                    'merged_from': [m['id'] for m in similar_memories]
                }
                
                consolidated.append(consolidated_memory)
                print(f"✓ Merged {len(similar_memories)} similar memories")
                print(f"  Result: '{merged_text[:50]}...'")
            else:
                consolidated.append(mem1)
            
            processed.add(i)
        
        # Update memory list
        old_count = len(self.memory)
        self.memory = consolidated
        new_count = len(self.memory)
        
        print(f"\n📊 Consolidation Summary:")
        print(f"   Before: {old_count} memories")
        print(f"   After: {new_count} memories")
        
        if old_count > 0:
            reduction = old_count - new_count
            reduction_percent = (reduction / old_count) * 100
            print(f"   Reduction: {reduction} memories ({reduction_percent:.1f}%)")
        
        self.save_memory()
    
    def merge_memories(self, memories):
        """Merge text from similar memories"""
        if not memories:
            return ""
        
        # Extract unique information from each memory
        all_sentences = []
        
        for mem in memories:
            text = mem['text'].strip()
            if text.endswith('.'):
                sentences = text.split('. ')
            else:
                sentences = [text]
            
            for sentence in sentences:
                if sentence and sentence not in all_sentences:
                    all_sentences.append(sentence)
        
        # Create merged text (limit to 3 most important sentences)
        if len(all_sentences) <= 3:
            result = '. '.join(all_sentences)
            if all_sentences and not result.endswith('.'):
                result += '.'
            return result
        else:
            # Sort by length (assuming longer sentences have more info)
            sorted_sentences = sorted(all_sentences, key=len, reverse=True)
            result = '. '.join(sorted_sentences[:3])
            if not result.endswith('.'):
                result += '.'
            return result
    
    def store_memory(self, text):
        """Store a new memory with automatic forgetting check"""
        if not text:
            print("❌ Cannot store empty memory")
            return None
        
        # Check if we need to make room
        if len(self.memory) >= self.max_memories:
            print("⚠️  Memory capacity reached. Applying forgetting...")
            self.apply_forgetting()
        
        keywords = self.extract_keywords(text)
        
        memory_entry = {
            'id': len(self.memory),
            'text': text,
            'keywords': keywords,
            'timestamp': self.time_step,
            'access_count': 0,
            'original_text': text.lower(),
            'creation_time': datetime.now().isoformat(),
            'last_accessed': None,
            'relevance_score': 1.0,  # Fresh memory starts with max relevance
            'merged_from': []  # Track if this was merged from other memories
        }
        
        self.memory.append(memory_entry)
        self.time_step += 1
        self.save_memory()
        
        print(f"✓ Stored: '{text}'")
        print(f"  Keywords: {keywords}")
        print(f"  Memory count: {len(self.memory)}/{self.max_memories}")
        
        return memory_entry
    
    def retrieve_memories(self, query, apply_forgetting_after=True):
        """Retrieve relevant memories for a query"""
        if not query:
            return []
        
        query_keywords = self.extract_keywords(query)
        query_text = query.lower()
        
        scored_memories = []
        
        for memory in self.memory:
            memory_keywords = set(memory['keywords'])
            memory_text = memory['original_text']
            
            score = 0
            
            # Method 1: Direct text match
            for keyword in query_keywords:
                if keyword and keyword in memory_text:
                    score += 3.0
            
            # Method 2: Keyword overlap
            overlap = set(query_keywords) & memory_keywords
            if overlap:
                score += len(overlap) * 2.0
            
            # Method 3: Relevance score boost
            relevance = memory.get('relevance_score', 0.5)
            score *= (0.5 + 0.5 * relevance)  # Memories with higher relevance get boost
            
            # Method 4: Age-based decay
            if score > 0:
                age = self.time_step - memory.get('timestamp', 0)
                decay_factor = math.exp(-self.decay_rate * age)
                score *= decay_factor
                
                # Update access statistics
                memory['access_count'] = memory.get('access_count', 0) + 1
                memory['last_accessed'] = self.time_step
                scored_memories.append((score, memory))
        
        # Sort by score (highest first)
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        
        # Update relevance scores
        for memory in self.memory:
            memory['relevance_score'] = self.calculate_memory_relevance(memory)
        
        # Apply forgetting if needed
        if apply_forgetting_after and len(scored_memories) > 0:
            # Check if any memories are below threshold
            low_relevance_count = sum(1 for mem in self.memory 
                                    if mem.get('relevance_score', 0) < self.forget_threshold)
            
            if low_relevance_count > 0:
                print(f"⚠️  {low_relevance_count} memories below relevance threshold")
        
        # Return top 3 memories
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
        
        # Log the query
        self.query_log.append({
            'query': query,
            'memories_found': len(memories),
            'response_time': response_time,
            'timestamp': time.time()
        })
        
        if not memories:
            return {
                'answer': f"I don't have information about '{query}'",
                'memories': [],
                'count': 0,
                'response_time': response_time
            }
        
        # Build answer from memories
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
                relevance = memory.get('relevance_score', 'N/A')
                relevance_str = f"{relevance:.3f}" if isinstance(relevance, (int, float)) else relevance
                
                print(f"  {i}. '{memory['text']}'")
                print(f"     Keywords: {', '.join(memory['keywords'][:3])}")
                print(f"     Accessed: {memory.get('access_count', 0)} times")
                print(f"     Relevance: {relevance_str}")
                if memory.get('merged_from'):
                    print(f"     Merged from: {len(memory['merged_from'])} memories")
        print("="*60)
    
    def show_memory_health(self):
        """Show memory health statistics"""
        print("\n" + "="*60)
        print("🧠 MEMORY HEALTH REPORT")
        print("="*60)
        
        if not self.memory:
            print("No memories stored yet.")
            return
        
        # Calculate statistics
        total_memories = len(self.memory)
        relevance_scores = [m.get('relevance_score', 0) for m in self.memory]
        access_counts = [m.get('access_count', 0) for m in self.memory]
        ages = [self.time_step - m.get('timestamp', 0) for m in self.memory]
        
        # Categories
        strong_memories = sum(1 for s in relevance_scores if s > 0.7)
        weak_memories = sum(1 for s in relevance_scores if s < 0.3)
        at_risk_memories = sum(1 for s in relevance_scores if s < self.forget_threshold)
        
        print(f"📊 Memory Statistics:")
        print(f"  Total memories: {total_memories}/{self.max_memories}")
        print(f"  Capacity usage: {(total_memories/self.max_memories)*100:.1f}%")
        print(f"  Strong memories (>0.7): {strong_memories}")
        print(f"  Weak memories (<0.3): {weak_memories}")
        print(f"  At-risk memories (<{self.forget_threshold}): {at_risk_memories}")
        
        print(f"\n📈 Relevance Distribution:")
        if relevance_scores:
            print(f"  Average relevance: {sum(relevance_scores)/len(relevance_scores):.3f}")
            print(f"  Min relevance: {min(relevance_scores):.3f}")
            print(f"  Max relevance: {max(relevance_scores):.3f}")
        
        print(f"\n👁️  Access Patterns:")
        if access_counts:
            print(f"  Total accesses: {sum(access_counts)}")
            print(f"  Average accesses: {sum(access_counts)/len(access_counts):.1f}")
            print(f"  Most accessed: {max(access_counts)} times")
            print(f"  Least accessed: {min(access_counts)} times")
        
        print(f"\n🕐 Age Distribution:")
        if ages:
            print(f"  Average age: {sum(ages)/len(ages):.0f} steps")
            print(f"  Oldest: {max(ages)} steps")
            print(f"  Newest: {min(ages)} steps")
        
        # Show memory categories
        print(f"\n📋 Memory Categories:")
        
        # Sort by relevance
        sorted_memories = sorted(self.memory, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        print(f"\n🏆 Top 3 Most Relevant Memories:")
        for i, mem in enumerate(sorted_memories[:3], 1):
            print(f"  {i}. '{mem['text'][:40]}...'")
            print(f"     Relevance: {mem.get('relevance_score', 0):.3f}, "
                  f"Accessed: {mem.get('access_count', 0)} times")
        
        print(f"\n⚠️  Bottom 3 Least Relevant Memories (at risk):")
        for i, mem in enumerate(sorted_memories[-3:], 1):
            print(f"  {i}. '{mem['text'][:40]}...'")
            print(f"     Relevance: {mem.get('relevance_score', 0):.3f}, "
                  f"Accessed: {mem.get('access_count', 0)} times")
        
        # Forgetting log summary
        if self.forgetting_log:
            recent_forgetting = self.forgetting_log[-5:]  # Last 5 forgotten memories
            print(f"\n🗑️  Recently Forgotten Memories (last 5):")
            for i, entry in enumerate(recent_forgetting, 1):
                print(f"  {i}. '{entry.get('text', '')[:30]}...'")
                print(f"     Reason: {entry.get('reason', 'unknown')}, "
                      f"Score: {entry.get('relevance_score', 0):.3f}")
        
        print("="*60)
    
    def run_forgetting_experiment(self):
        """Run forgetting mechanism experiment"""
        print("\n" + "="*80)
        print("🧪 FORGETTING MECHANISM EXPERIMENT")
        print("="*80)
        
        # Store initial state
        initial_count = len(self.memory)
        
        print("\n📝 Starting experiment...")
        print(f"Initial memory count: {initial_count}")
        print(f"Forget threshold: {self.forget_threshold}")
        print(f"Max memories: {self.max_memories}")
        
        # Step 1: Apply forgetting
        print("\n1️⃣  Applying forgetting mechanism...")
        forgotten = self.apply_forgetting()
        
        # Step 2: Consolidate memories
        print("\n2️⃣  Consolidating memories...")
        self.consolidate_memories()
        
        # Step 3: Show results
        print("\n3️⃣  Experiment Results:")
        print("-"*40)
        
        final_count = len(self.memory)
        forgotten_count = initial_count - final_count
        
        print(f"Memories before: {initial_count}")
        print(f"Memories after: {final_count}")
        print(f"Memories forgotten: {forgotten_count}")
        
        if initial_count > 0:
            print(f"Reduction: {(forgotten_count/initial_count)*100:.1f}%")
        
        # Show health report
        self.show_memory_health()
        
        # Save experiment results
        with open("forgetting_experiment_results.txt", "w") as f:
            f.write("Forgetting Mechanism Experiment Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Initial memories: {initial_count}\n")
            f.write(f"Final memories: {final_count}\n")
            f.write(f"Memories forgotten: {forgotten_count}\n")
            if initial_count > 0:
                f.write(f"Reduction rate: {(forgotten_count/initial_count)*100:.1f}%\n")
            f.write(f"Forget threshold: {self.forget_threshold}\n")
            f.write(f"Max capacity: {self.max_memories}\n\n")
            
            if forgotten:
                f.write("Forgotten Memories:\n")
                f.write("-"*30 + "\n")
                for mem in forgotten:
                    f.write(f"- {mem['text']}\n")
                    f.write(f"  Relevance: {mem.get('relevance_score', 0):.3f}\n")
                    f.write(f"  Age: {self.time_step - mem.get('timestamp', 0)} steps\n\n")
            else:
                f.write("No memories were forgotten.\n")
        
        print(f"\n📄 Experiment results saved to 'forgetting_experiment_results.txt'")
        print("="*80)


def main():
    """Main program with enhanced forgetting features"""
    print("\n" + "="*60)
    print("🤖 ADVANCED MEMORY AGENT WITH FORGETTING MECHANISM")
    print("="*60)
    
    # Initialize advanced agent
    agent = AdvancedMemoryAgent(
        decay_rate=0.02,
        forget_threshold=0.2,  # Forget memories below 20% relevance
        max_memories=20  # Maximum 20 memories before pruning
    )
    
    # Check if we need to store initial memories
    if len(agent.memory) == 0:
        print("\n📝 SETTING UP INITIAL MEMORIES...")
        print("-"*40)
        
        initial_memories = [
            "My favorite sport is football and I play it every weekend.",
            "I study artificial intelligence and machine learning at university.",
            "Memory decay is important for long-term reasoning in AI systems.",
            "I want to build advanced AGI systems that can reason like humans.",
            "Python is my primary programming language for AI development.",
            "Deep learning models require large amounts of training data.",
            "Reinforcement learning agents learn through trial and error.",
            "Natural language processing helps computers understand human language.",
            "I enjoy watching basketball games on television.",
            "Machine learning algorithms can recognize patterns in data.",
            "Neural networks are inspired by the human brain structure.",
            "I prefer coffee over tea in the morning.",
            "Data preprocessing is crucial for machine learning success.",
            "I visited Paris last summer for a research conference.",
            "Transfer learning allows models to use knowledge from one task for another.",
            "Regularization techniques prevent overfitting in machine learning.",
            "I like to read science fiction books in my free time.",
            "Computer vision enables machines to interpret visual information.",
            "Unsupervised learning finds patterns without labeled data.",
            "I believe AI will transform healthcare in the next decade."
        ]
        
        for text in initial_memories:
            agent.store_memory(text)
            time.sleep(0.1)  # Small delay to simulate time passing
    
    else:
        print(f"\n✅ Found {len(agent.memory)} existing memories")
        print("You can start querying right away!")
    
    # Main menu loop
    while True:
        print("\n" + "="*60)
        print("ADVANCED MEMORY AGENT MENU")
        print("="*60)
        print("1. Test a query")
        print("2. Store new memory")
        print("3. Run forgetting experiment")
        print("4. Show memory health report")
        print("5. Apply forgetting mechanism now")
        print("6. Consolidate memories (merge similar)")
        print("7. View all memories with relevance scores")
        print("8. Show forgetting history")
        print("9. Run research analysis")
        print("10. Exit")
        print("="*60)
        
        choice = input("Select option (1-10): ").strip()
        
        if choice == '1':
            # Test a query
            print("\n" + "="*60)
            print("TEST QUERY")
            print("="*60)
            query = input("Enter your query: ").strip()
            
            if query:
                result = agent.answer_query(query)
                agent.display_result(result, query)
            else:
                print("❌ Please enter a query")
        
        elif choice == '2':
            # Store new memory
            print("\n" + "="*60)
            print("STORE NEW MEMORY")
            print("="*60)
            text = input("Enter memory text: ").strip()
            
            if text:
                agent.store_memory(text)
                print("✅ Memory stored successfully!")
            else:
                print("❌ Please enter some text")
        
        elif choice == '3':
            # Run forgetting experiment
            agent.run_forgetting_experiment()
        
        elif choice == '4':
            # Show memory health report
            agent.show_memory_health()
        
        elif choice == '5':
            # Apply forgetting now
            print("\n" + "="*60)
            print("APPLY FORGETTING NOW")
            print("="*60)
            confirm = input("Apply forgetting mechanism? (yes/no): ").strip().lower()
            
            if confirm == 'yes':
                forgotten = agent.apply_forgetting()
                if forgotten:
                    print(f"\n✅ Forgot {len(forgotten)} memories")
                else:
                    print("\n✅ No memories needed forgetting")
            else:
                print("❌ Cancelled")
        
        elif choice == '6':
            # Consolidate memories - NOW FIXED!
            print("\n" + "="*60)
            print("CONSOLIDATE MEMORIES")
            print("="*60)
            confirm = input("Merge similar memories? (yes/no): ").strip().lower()
            
            if confirm == 'yes':
                agent.consolidate_memories()
                print("✅ Memory consolidation complete!")
            else:
                print("❌ Cancelled")
        
        elif choice == '7':
            # View all memories with relevance
            print("\n" + "="*60)
            print("ALL MEMORIES WITH RELEVANCE SCORES")
            print("="*60)
            
            if agent.memory:
                # Sort by relevance
                sorted_memories = sorted(agent.memory, 
                                       key=lambda x: x.get('relevance_score', 0), 
                                       reverse=True)
                
                for i, mem in enumerate(sorted_memories, 1):
                    relevance = mem.get('relevance_score', 'N/A')
                    relevance_str = f"{relevance:.3f}" if isinstance(relevance, (int, float)) else relevance
                    
                    status = "✅" if relevance > agent.forget_threshold else "⚠️ "
                    
                    print(f"{status} {i}. '{mem['text']}'")
                    print(f"   Relevance: {relevance_str}, "
                          f"Accessed: {mem.get('access_count', 0)} times, "
                          f"Age: {agent.time_step - mem.get('timestamp', 0)} steps")
                    if mem.get('merged_from'):
                        print(f"   Merged from: {len(mem['merged_from'])} memories")
                    print()
            else:
                print("No memories stored yet.")
        
        elif choice == '8':
            # Show forgetting history
            print("\n" + "="*60)
            print("FORGETTING HISTORY")
            print("="*60)
            
            if agent.forgetting_log:
                print(f"Total memories forgotten: {len(agent.forgetting_log)}")
                print("\nRecent forgetting events:")
                
                # Show last 10
                for i, entry in enumerate(agent.forgetting_log[-10:], 1):
                    print(f"\n{i}. '{entry.get('text', '')[:50]}...'")
                    print(f"   Reason: {entry.get('reason', 'unknown')}")
                    print(f"   Score: {entry.get('relevance_score', 0):.3f}")
                    print(f"   When: {entry.get('forgotten_at', 'unknown')[:19]}")
            else:
                print("No memories have been forgotten yet.")
        
        elif choice == '9':
            # Run research analysis (simplified version)
            print("\n" + "="*60)
            print("RESEARCH ANALYSIS")
            print("="*60)
            
            test_queries = [
                "What sport do I like?",
                "What do I study?",
                "Tell me about machine learning",
                "What programming language?",
                "Random query with no match"
            ]
            
            print("\n📊 Query Performance:")
            print("-"*40)
            
            for query in test_queries:
                result = agent.answer_query(query)
                print(f"'{query[:20]}...': {result['count']} memories found")
        
        elif choice == '10':
            # Exit
            print("\n" + "="*60)
            print("EXITING PROGRAM")
            print("="*60)
            print("Saving memories...")
            agent.save_memory()
            agent.save_forgetting_log()
            print(f"✅ Saved {len(agent.memory)} memories")
            print(f"✅ Saved {len(agent.forgetting_log)} forgetting records")
            print("\nThank you for using the Advanced Memory Agent!")
            print("="*60)
            break
        
        else:
            print("❌ Invalid choice. Please select 1-10.")
        
        # Pause before showing menu again
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()