import math
import random
import string
from datetime import datetime
from collections import Counter
from .logger import FileHandler
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer


class AIMemory:
    def __init__(self, file_handler: FileHandler, decay_rate=0.01, forget_threshold=0.1, max_memories=50):
        self.file_handler = file_handler
        self.decay_rate = decay_rate
        self.forget_threshold = forget_threshold
        self.max_memories = max_memories
        self.memory = file_handler.memory
        self.forgetting_log = file_handler.forgetting_log
        self.time_step = file_handler.time_step
        self.lemmatizer = WordNetLemmatizer()
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words("english")) - {"not", "no", "never"}
        self.memory_categories = {}

    def extract_keywords(self, text, min_keywords=3, max_keywords=6):
        if not text:
            return []
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        candidates = []
        for word, tag in tagged:
            if word in self.stop_words or word in string.punctuation:
                continue
            if len(word) < 3 or not word.isalpha():
                continue
            if tag not in {"NN", "NNS", "NNP", "NNPS"}:
                continue
            if abs(self.sia.polarity_scores(word)["compound"]) > 0.3:
                continue
            candidates.append(self.lemmatizer.lemmatize(word))
        if not candidates:
            return []
        ranked = [w for w, _ in Counter(candidates).most_common()]
        return ranked[:max_keywords] if len(ranked) >= min_keywords else ranked

    def analyze_text_meaningfulness(self, text):
        keywords = self.extract_keywords(text)
        all_words = text.lower().split()
        cleaned_words = [w.strip('.,!?;:"\'()[]{}') for w in all_words if len(w.strip('.,!?;:"\'()[]{}')) > 1]
        meaningful_score = len(keywords) / len(cleaned_words) if keywords and cleaned_words else 0.0
        return keywords, meaningful_score

    def find_similar_memories(self, text, keywords, threshold=0.7):
        similar = []
        for memory in self.memory:
            memory_keywords = set(memory['keywords'])
            new_keywords = set(keywords)
            if not memory_keywords or not new_keywords:
                continue
            similarity = len(memory_keywords & new_keywords) / len(memory_keywords | new_keywords)
            if similarity > threshold:
                similar.append((similarity, memory))
        return similar

    def merge_memories(self, text, keywords, similar_memories):
        memory_texts = [mem['text'] for _, mem in similar_memories] + [text]
        all_sentences = []
        for text_item in memory_texts:
            all_sentences.extend(sent_tokenize(text_item))
        seen = set()
        unique_sentences = []
        for sentence in all_sentences:
            clean = sentence.strip().lower()
            if clean and clean not in seen:
                seen.add(clean)
                unique_sentences.append(sentence)
        all_tokens = []
        for text_item in memory_texts:
            tokens = word_tokenize(text_item.lower())
            all_tokens.extend([t for t in tokens if t.isalnum()])
        word_freq = Counter(all_tokens)
        scored_sentences = []
        for sentence in unique_sentences:
            score = sum(1.0 / (word_freq.get(t.lower(), 0) + 1) for t in word_tokenize(sentence.lower()))
            scored_sentences.append((score, sentence))
        scored_sentences.sort(reverse=True)
        merged_text = " ".join([s for _, s in scored_sentences[:min(5, len(scored_sentences))]])
        new_keywords = self.extract_keywords(merged_text)
        max_relevance = max([mem.get('relevance_score', 0) for _, mem in similar_memories] + [1.0])
        memory_ids_to_remove = [mem['id'] for _, mem in similar_memories]
        self.memory = [mem for mem in self.memory if mem['id'] not in memory_ids_to_remove]
        merged_memory = {
            'id': len(self.memory),
            'text': merged_text,
            'keywords': new_keywords,
            'timestamp': self.time_step,
            'access_count': 1,
            'relevance_score': max_relevance,
            'meaningfulness_score': 1.0,
            'merged_from': memory_ids_to_remove,
            'is_merged': True,
            'last_accessed': None
        }
        self.memory.append(merged_memory)
        self.file_handler.save_memory()
        return merged_memory

    def store_memory(self, text):
        if not text:
            return None
        keywords, meaningful_score = self.analyze_text_meaningfulness(text)
        if not keywords:
            return None
        if meaningful_score < 0.1:
            return None
        similar_memories = self.find_similar_memories(text, keywords, 0.6)
        if similar_memories:
            merged = self.merge_memories(text, keywords, similar_memories)
            return merged
        if len(self.memory) >= self.max_memories:
            self.apply_forgetting()
        memory_entry = {
            'id': len(self.memory),
            'text': text,
            'keywords': keywords,
            'timestamp': self.time_step,
            'access_count': 0,
            'relevance_score': 1.0,
            'meaningfulness_score': meaningful_score,
            'merged_from': [],
            'last_accessed': None
        }
        self.memory.append(memory_entry)
        self.time_step += 1
        self.file_handler.time_step = self.time_step
        self.file_handler.save_memory()
        return memory_entry

    def calculate_memory_relevance(self, memory):
        age = self.time_step - memory.get('timestamp', 0)
        score = math.exp(-self.decay_rate * age)
        access_factor = min(memory.get('access_count', 0) / 10, 1.0)
        score *= (0.3 + 0.7 * access_factor)
        if memory.get('last_accessed'):
            last_access_age = self.time_step - memory['last_accessed']
            score *= math.exp(-0.1 * last_access_age)
        return max(0.0, min(1.0, score))

    def intelligent_forgetting(self, candidate_memories):
        if len(candidate_memories) < 2:
            return candidate_memories[:2]
        memory_scores = []
        for memory in candidate_memories:
            full_memory = next((m for m in self.memory if m.get('id') == memory.get('id')), memory)
            relevance = self.calculate_memory_relevance(full_memory)
            memory_scores.append({'memory': memory, 'relevance': relevance})
        memory_scores.sort(key=lambda x: x['relevance'])
        num_to_select = 3 if len(memory_scores) >= 3 else 2
        return [item['memory'] for item in memory_scores[:num_to_select]]

    def assess_memory_quality(self, memory_text):
        if not memory_text or not memory_text.strip():
            return None
        temp_memory = {'text': memory_text, 'timestamp': self.time_step, 'access_count': 0}
        base_relevance = self.calculate_memory_relevance(temp_memory)
        words = memory_text.split()
        word_count = len(words)
        lexical_diversity = len(set(words)) / max(word_count, 1)
        emotional_words = {'love', 'happy', 'sad', 'angry', 'scared', 'excited', 'worried'}
        emotional_count = sum(1 for word in words if word.lower() in emotional_words)
        emotional_value = min(emotional_count / max(word_count, 1) * 20, 10)
        clarity = min(max(word_count / 15, 1), 10)
        usefulness = min(base_relevance * 10, 10)
        uniqueness = min(lexical_diversity * 10, 10)
        emotional_val = min(emotional_value, 10)
        detail = min((word_count / 50) * 10, 10)
        return {
            "clarity": round(clarity, 1),
            "usefulness": round(usefulness, 1),
            "uniqueness": round(uniqueness, 1),
            "emotional_value": round(emotional_val, 1),
            "detail": round(detail, 1)
        }

    def generate_memory_timeline(self):
        if len(self.memory) < 3:
            return []
        sorted_mems = sorted(self.memory, key=lambda m: m.get("timestamp", 0))[-8:]
        timeline = []
        for i, mem in enumerate(sorted_mems):
            text = mem.get('text', '')
            if len(text.split()) > 10:
                words = text.split()
                summary = ' '.join(words[:8] + ['...'] + words[-2:]) if len(words) > 10 else text
            else:
                summary = text
            timeline.append({
                'text': summary,
                'timestamp': mem.get('timestamp', 0),
                'original_text': text,
                'time_index': i,
                'relative_time': f"Memory {i+1}/{len(sorted_mems)}"
            })
        return timeline

    def apply_forgetting(self):
        for memory in self.memory:
            memory['relevance_score'] = self.calculate_memory_relevance(memory)
        self.memory.sort(key=lambda x: x['relevance_score'])
        memories_to_forget = []
        for memory in self.memory:
            if memory['relevance_score'] < self.forget_threshold or (len(self.memory) > self.max_memories and memory['relevance_score'] < 0.5):
                memories_to_forget.append(memory)
        if len(memories_to_forget) > 3:
            memories_to_forget = self.intelligent_forgetting(memories_to_forget)
        memory_ids_to_forget = [mem['id'] for mem in memories_to_forget]
        for memory in memories_to_forget:
            forget_entry = {
                'memory_id': memory['id'],
                'text': memory['text'],
                'relevance_score': memory['relevance_score'],
                'forgotten_at': datetime.now().isoformat(),
                'reason': 'low_relevance' if memory['relevance_score'] < self.forget_threshold else 'capacity_limit',
            }
            self.forgetting_log.append(forget_entry)
        self.memory = [mem for mem in self.memory if mem['id'] not in memory_ids_to_forget]
        self.file_handler.save_memory()
        self.file_handler.save_forgetting_log()
        return memories_to_forget

    def retrieve_memories(self, query):
        if not query:
            return []
        query_keywords = self.extract_keywords(query)
        scored_memories = []
        for memory in self.memory:
            memory_keywords = set(memory['keywords'])
            overlap = set(query_keywords) & memory_keywords
            score = len(overlap) * 2.0 if overlap else 0
            score *= (0.5 + 0.5 * memory.get('relevance_score', 0.5))
            if score > 0:
                memory['access_count'] = memory.get('access_count', 0) + 1
                memory['last_accessed'] = self.time_step
                scored_memories.append((score, memory))
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        return [memory for score, memory in scored_memories[:3]]

    def answer_query(self, query):
        if not query:
            return {'answer': "Please enter a query.", 'memories': [], 'count': 0}
        memories = self.retrieve_memories(query)
        if not memories:
            return {'answer': f"I don't have information about '{query}'", 'memories': [], 'count': 0}
        memory_texts = [mem['text'] for mem in memories]
        if len(memory_texts) == 1:
            answer = f"I remember: {memory_texts[0]}"
        else:
            answer = "Based on my memories:\n" + "\n".join(f"{i}. {text}" for i, text in enumerate(memory_texts, 1))
        return {'answer': answer.strip(), 'memories': memories, 'count': len(memories)}

    def discover_memory_relationships(self):
        if len(self.memory) < 2:
            return ""
        sample = random.sample(self.memory, min(5, len(self.memory)))
        texts = [mem['text'] for mem in sample]
        all_keywords = []
        for text in texts:
            all_keywords.extend(self.extract_keywords(text))
        keyword_freq = Counter(all_keywords)
        common_keywords = [kw for kw, freq in keyword_freq.items() if freq > 1]
        relationships = []
        if common_keywords:
            relationships.append(f"Common themes: {', '.join(common_keywords[:5])}")
        for i, mem1 in enumerate(sample):
            for j, mem2 in enumerate(sample[i+1:], i+1):
                kw1, kw2 = set(mem1['keywords']), set(mem2['keywords'])
                if kw1 and kw2:
                    overlap = kw1 & kw2
                    if overlap:
                        relationships.append(f"Memory {i+1} & {j+1} share: {', '.join(list(overlap)[:3])}")
        return "\n".join(relationships[:5]) if relationships else "No strong relationships found"

    def auto_categorize_memories(self):
        categories = {}
        category_keywords = {
            'Personal': {'i', 'my', 'me', 'friend', 'family', 'home'},
            'Technical': {'code', 'python', 'function', 'system', 'data'},
            'Facts': {'fact', 'know', 'information', 'true', 'data'},
            'Goals': {'goal', 'want', 'need', 'plan', 'future'},
            'Trivia': {'interesting', 'fun', 'random', 'factoid'},
            'Other': set()
        }
        for memory in self.memory:
            if 'category' in memory:
                continue
            memory_keywords = set(memory['keywords'])
            best_category = 'Other'
            best_score = 0
            for category, keywords in category_keywords.items():
                if category == 'Other':
                    continue
                overlap = memory_keywords & keywords
                score = len(overlap)
                if score > best_score:
                    best_score = score
                    best_category = category
            memory['category'] = best_category
            categories.setdefault(best_category, []).append(memory['id'])
        self.memory_categories = categories
        self.file_handler.save_memory()
        return categories