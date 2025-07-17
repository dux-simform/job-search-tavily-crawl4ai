#!/usr/bin/env python3
"""
Simple example script showing how to use the job search engine.
"""

import asyncio
from job_search_engine import search_and_extract_jobs

async def search_jobs_example():
    """Example of searching for jobs with user input"""
    
    print("🤖 Job Search Engine Demo")
    print("=" * 40)
    
    # Get user input
    user_query = input("Enter your job search query (e.g., 'Python developer remote'): ").strip()
    
    if not user_query:
        user_query = "Python developer remote"  # Default query
        print(f"Using default query: {user_query}")
    
    # Ask for number of jobs to return in final results
    try:
        max_jobs = int(input("How many jobs do you want in the final results? (default: all jobs found): ") or "0")
        if max_jobs <= 0:
            max_jobs = None  # Return all jobs
    except ValueError:
        max_jobs = None  # Return all jobs
    
    print("✨ Job enhancement is enabled by default - will crawl job application pages for comprehensive details")
    
    print(f"\n🔍 Searching for: '{user_query}'")
    if max_jobs:
        print(f"📊 Will return up to {max_jobs} jobs in final results")
    else:
        print(f"📊 Will return all jobs found")
    print("-" * 50)
    
    # Search for jobs
    try:
        results = await search_and_extract_jobs(
            user_query=user_query,
            days=7,  # Look for jobs posted in the last 7 days
            max_results=5,  # Number of URLs to search (default)
            max_jobs_to_return=max_jobs  # Number of jobs to return
        )
        
        # Display results
        print(f"\n🎯 Found {results.total_jobs_found} jobs!")
        
        print("\n💾 Results automatically saved to output/ directory:")
        print(f"   • Enhanced job results: output/search_results/")
        
        if results.raw_markdown_files:
            print(f"   • Raw crawled content ({len(results.raw_markdown_files)} files):")
            print(f"     - Main search: output/raw_crawled/")
            print(f"     - Enhanced crawl: output/enhanced_crawled/")
            for raw_file in results.raw_markdown_files:
                print(f"     - {raw_file}")
        
        # Summary
        print("\n📈 Summary:")
        print(f"   • Found {results.total_jobs_found} jobs")
        print(f"   • Enhanced {results.enhanced_jobs_count} jobs with additional details")
        print(f"   • Failed to process {len(results.failed_urls)} URLs")
        print(f"   • Saved {len(results.raw_markdown_files)} raw content files")
        print(f"   • All results saved in organized output/ directory structure")
        
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
        print("2. AZURE_OPENAI_API_KEY set in your .env file")
        print("3. Required packages installed (crawl4ai, tavily-python, python-dotenv)")
        print("4. Internet connection for API calls")

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
                    days=7,
                    max_results=5,  # Search 5 URLs for examples
                    max_jobs_to_return=10  # Limit to 10 jobs for examples
                )
                
                print(f"🎯 Found {results.total_jobs_found} jobs for '{query}'")
                print(f"📁 Results automatically saved to output/ directory")
                if results.enhanced_jobs_count > 0:
                    print(f"✨ Enhanced {results.enhanced_jobs_count} jobs with additional details")
                if results.raw_markdown_files:
                    print(f"📄 Raw content files: {len(results.raw_markdown_files)} files saved in output/")
                
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
