from ai import FileHandler
from ai import AIMemory

def initialize_memories(agent: AIMemory):
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
    ]
    for text in initial_memories:
        agent.store_memory(text)
    print(f"✅ Initialized {len(initial_memories)} memories")

def handle_test_query(agent: AIMemory):
    print("\n" + "="*60)
    print("TEST QUERY")
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
        if result['memories']:
            print("\n🔍 Retrieved memories:")
            for i, memory in enumerate(result['memories'], 1):
                print(f"  {i}. '{memory['text']}'")
                print(f"     Keywords: {', '.join(memory['keywords'][:3])}")
        print("="*60)

def handle_store_memory(agent: AIMemory):
    print("\n" + "="*60)
    print("STORE NEW MEMORY")
    print("="*60)
    text = input("Enter memory text: ").strip()
    if text:
        stored = agent.store_memory(text)
        print(f"✅ Memory stored: {stored['text'][:50]}..." if stored else "❌ Memory not stored")

def handle_forgetting(agent: AIMemory):
    print("\n" + "="*60)
    print("APPLY FORGETTING")
    print("="*60)
    confirm = input("Apply forgetting mechanism? (yes/no): ").strip().lower()
    if confirm == 'yes':
        forgotten = agent.apply_forgetting()
        print(f"✅ Forgot {len(forgotten)} memories")

def handle_view_memories(agent: AIMemory):
    print("\n" + "="*60)
    print("ALL MEMORIES")
    print("="*60)
    if agent.memory:
        for i, mem in enumerate(agent.memory, 1):
            memory_type = " [MERGED]" if mem.get('is_merged') else ""
            print(f"{i}. '{mem['text']}'{memory_type}")
            print(f"   Keywords: {mem['keywords']}")
            print(f"   Relevance: {mem.get('relevance_score', 0):.2f}")
            print()
    else:
        print("No memories stored.")

def handle_keyword_extraction(agent: AIMemory):
    print("\n" + "="*60)
    print("TEST KEYWORD EXTRACTION")
    print("="*60)
    text = input("Enter text to analyze: ").strip()
    if text:
        keywords, score = agent.analyze_text_meaningfulness(text)
        print(f"\n📊 Results:")
        print(f"   Keywords: {keywords}")
        print(f"   Score: {score:.2f}")

def handle_pattern_discovery(agent: AIMemory):
    print("\n" + "="*60)
    print("DISCOVER PATTERNS")
    print("="*60)
    analysis = agent.discover_memory_relationships()
    print(f"\n🔍 Pattern Analysis:")
    print(analysis if analysis else "No strong patterns found")

def handle_categorize_memories(agent: AIMemory):
    print("\n" + "="*60)
    print("AUTO-CATEGORIZE MEMORIES")
    print("="*60)
    categories = agent.auto_categorize_memories()
    print(f"\n✅ Categorized memories into {len(categories)} categories")
    for category, ids in categories.items():
        print(f"   {category}: {len(ids)} memories")

def handle_assess_quality(agent: AIMemory):
    print("\n" + "="*60)
    print("ASSESS MEMORY QUALITY")
    print("="*60)
    text = input("Enter memory text to assess: ").strip()
    if text:
        assessment = agent.assess_memory_quality(text)
        print(f"\n📊 Quality Assessment:")
        for key, value in assessment.items():
            print(f"   {key}: {value}")

def handle_generate_timeline(agent: AIMemory):
    print("\n" + "="*60)
    print("GENERATE TIMELINE")
    print("="*60)
    timeline = agent.generate_memory_timeline()
    print(f"\n📅 Memory Timeline:")
    for item in timeline:
        print(f"  {item['text']}")

def handle_export_memories(agent: AIMemory):
    print("\n" + "="*60)
    print("EXPORT MEMORIES")
    print("="*60)
    filename = input("Export filename (default: memory_export.json): ").strip()
    filename = filename if filename else "memory_export.json"
    agent.file_handler.export_memories(filename)

def show_memory_health(agent: AIMemory):
    print("\n🧠 MEMORY HEALTH REPORT")
    print("="*60)
    total = len(agent.memory)
    print(f"📊 Total memories: {total}/{agent.max_memories}")
    if total > 0:
        relevance_scores = [m.get('relevance_score', 0) for m in agent.memory]
        access_counts = [m.get('access_count', 0) for m in agent.memory]
        print(f"📉 Average relevance: {sum(relevance_scores)/total:.3f}")
        print(f"📈 Average access count: {sum(access_counts)/total:.1f}")
        merged_count = sum(1 for m in agent.memory if m.get('is_merged', False))
        print(f"📋 Merged memories: {merged_count}")
        if agent.memory_categories:
            print(f"🏷️  Categories:")
            for category, ids in agent.memory_categories.items():
                print(f"   {category}: {len(ids)}")
    print("="*60)

def show_memory_categories(agent: AIMemory):
    if not agent.memory_categories:
        print("\n⚠️  No categories assigned yet. Run auto-categorization first.")
        return
    print("\n🏷️  MEMORY CATEGORIES")
    print("="*60)
    for category, memory_ids in agent.memory_categories.items():
        print(f"\n{category.upper()}: {len(memory_ids)} memories")
        for mem_id in memory_ids[:3]:
            memory = next((m for m in agent.memory if m['id'] == mem_id), None)
            if memory:
                print(f"  - '{memory['text'][:50]}...'")
    print("="*60)

def main():
    print("\n" + "="*60)
    print("🤖 ENHANCED SMART MEMORY AGENT")
    print("="*60)
    
    file_handler = FileHandler()
    file_handler.load_memory()
    file_handler.load_forgetting_log()
    
    agent = AIMemory(
        file_handler=file_handler,
        decay_rate=0.02,
        forget_threshold=0.2,
        max_memories=25
    )
    
    if len(agent.memory) == 0:
        initialize_memories(agent)
    else:
        print(f"\n✅ Found {len(agent.memory)} existing memories")
    
    menu_options = {
        '1': ("Test a query", handle_test_query),
        '2': ("Store new memory", handle_store_memory),
        '3': ("Apply forgetting mechanism", handle_forgetting),
        '4': ("Show memory health report", lambda a: show_memory_health(a)),
        '5': ("View all memories", handle_view_memories),
        '6': ("Test keyword extraction", handle_keyword_extraction),
        '7': ("Discover patterns", handle_pattern_discovery),
        '8': ("Auto-categorize memories", handle_categorize_memories),
        '9': ("Assess memory quality", handle_assess_quality),
        '10': ("Generate timeline", handle_generate_timeline),
        '11': ("View memory categories", lambda a: show_memory_categories(a)),
        '12': ("Export memories", handle_export_memories),
        '13': ("Exit", None)
    }
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU - ENHANCED SMART MEMORY AGENT")
        print("="*60)
        for key, (description, _) in menu_options.items():
            print(f"{key}. {description}")
        print("="*60)
        
        choice = input("Select option (1-13): ").strip()
        
        if choice == '13':
            print("\n" + "="*60)
            print("EXITING")
            print("="*60)
            file_handler.save_memory()
            file_handler.save_forgetting_log()
            print("✅ All data saved")
            break
        
        if choice in menu_options:
            _, handler = menu_options[choice]
            if handler:
                handler(agent)
        else:
            print("❌ Invalid choice")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()