#!/usr/bin/env python3
"""
Simple example script showing how to use the job search engine.
"""

import asyncio
from job_search_engine import search_and_extract_jobs, format_job_results

async def search_jobs_example():
    """Example of searching for jobs with user input"""
    
    print("🤖 Job Search Engine Demo")
    print("=" * 40)
    
    # Get user input
    user_query = input("Enter your job search query (e.g., 'Python developer remote'): ").strip()
    
    if not user_query:
        user_query = "Python developer remote"  # Default query
        print(f"Using default query: {user_query}")
    
    # Optional: Get number of results
    try:
        max_results = int(input("How many job URLs to search? (default 5): ") or "5")
    except ValueError:
        max_results = 5
    
    print(f"\n🔍 Searching for: '{user_query}'")
    print(f"📊 Looking for up to {max_results} jobs...")
    print("-" * 50)
    
    # Search for jobs
    try:
        results = await search_and_extract_jobs(
            user_query=user_query,
            max_results=max_results,
            days=7  # Look for jobs posted in the last 7 days
        )
        
        # Display results
        print("\n" + format_job_results(results, format_type="detailed"))
        
        # Save results to file
        filename = f"search_results_{user_query.replace(' ', '_').lower()}.json"
        with open(filename, "w") as f:
            f.write(results.model_dump_json(indent=2))
        
        print(f"\n💾 Results saved to: {filename}")
        
        # Summary
        print("\n📈 Summary:")
        print(f"   • Found {results.total_jobs_found} jobs")
        print(f"   • Failed to process {len(results.failed_urls)} URLs")
        
        if results.jobs:
            companies = list(set(job.company_name for job in results.jobs if job.company_name))
            locations = list(set(job.location for job in results.jobs if job.location))
            print(f"   • Companies: {', '.join(companies[:5])}")
            if len(companies) > 5:
                print(f"     ... and {len(companies) - 5} more")
            print(f"   • Locations: {', '.join(locations[:3])}")
            if len(locations) > 3:
                print(f"     ... and {len(locations) - 3} more")
        
    except Exception as e:
        print(f"❌ Error during job search: {e}")
        print("Make sure you have:")
        print("1. TAVILY_API_KEY set in your .env file")
        print("2. Ollama running with qwen2:1.5b model")
        print("3. Required packages installed (crawl4ai, tavily-python, python-dotenv)")

def quick_search_examples():
    """Run some predefined searches for testing"""
    
    example_queries = [
        "Senior Python Developer",
        "DevOps Engineer AWS",
        "Data Scientist remote",
        "Frontend React Developer",
        "Machine Learning Engineer"
    ]
    
    print("🚀 Running example searches...")
    
    async def run_examples():
        for query in example_queries:
            print(f"\n{'='*60}")
            print(f"Searching: {query}")
            print('='*60)
            
            try:
                results = await search_and_extract_jobs(
                    user_query=query,
                    max_results=3,  # Limit for demo
                    days=7
                )
                
                print(format_job_results(results, format_type="summary"))
                
                # Brief pause between searches
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"❌ Error searching for '{query}': {e}")
    
    asyncio.run(run_examples())

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--examples":
        quick_search_examples()
    else:
        asyncio.run(search_jobs_example())
