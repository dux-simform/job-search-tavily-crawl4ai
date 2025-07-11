import os
import asyncio
from typing import List, Optional
from tavily import TavilyClient
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

class JobDetails(BaseModel):
    job_title: str = Field(..., description="Exact title of the job position")
    company_name: str = Field(..., description="Name of the hiring company or organization")
    location: Optional[str] = Field(None, description="Job location including city, state, country, or if remote/hybrid")
    salary_range: Optional[str] = Field(None, description="Salary range, hourly rate, or compensation package")
    job_type: Optional[str] = Field(None, description="Employment type: full-time, part-time, contract, internship, freelance")
    experience_level: Optional[str] = Field(None, description="Required experience level: entry-level, junior, mid-level, senior, lead, executive")
    description: Optional[str] = Field(None, description="Concise job description highlighting main responsibilities and role purpose")
    requirements: Optional[List[str]] = Field(None, description="Key skills, qualifications, technologies, and requirements needed")
    benefits: Optional[List[str]] = Field(None, description="Benefits, perks, and additional compensation offered")
    posted_date: Optional[str] = Field(None, description="When the job was posted or last updated")
    apply_url: Optional[str] = Field(None, description="Direct URL or application link for this job")
    source_url: str = Field(..., description="Original URL where this job was found")

class JobSearchResult(BaseModel):
    query: str = Field(..., description="Original search query used")
    total_jobs_found: int = Field(..., description="Total number of jobs successfully extracted")
    jobs: List[JobDetails] = Field(..., description="List of job details extracted")
    failed_urls: List[str] = Field(default_factory=list, description="URLs that failed to be processed")

async def search_and_extract_jobs(
    user_query: str,
    max_results: int = 10,
    days: int = 7,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None
) -> JobSearchResult:
    """
    Search for job postings using Tavily and extract detailed information using Crawl4AI.
    
    Args:
        user_query: User's job search query (e.g., "DevOps Engineer with 5 years experience")
        max_results: Maximum number of job URLs to search for
        days: Number of days to look back for recent postings
        include_domains: List of domains to specifically include in search
        exclude_domains: List of domains to exclude from search
    
    Returns:
        JobSearchResult containing extracted job details
    """
    
    # Default domains if none specified
    if include_domains is None:
        include_domains = [
            # "linkedin.com", 
            # "indeed.com", 
            # "glassdoor.com", 
            # "ziprecruiter.com", 
            "ycombinator.com",
            # "angel.co",
            # "stackoverflow.com/jobs",
            # "dice.com",
            # "monster.com"
        ]
    
    # Initialize Tavily client
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    # Optimized search query for job postings
    search_query = f"""
    Find recent job posting URLs for: {user_query}
    Look for actual job listings and career pages with specific job openings.
    Include direct links to job applications and detailed job descriptions.
    Focus on active, current job postings from the last {days} days.
    """
    # search_query = f"recent job openings {user_query} hiring apply"
    
    print(f"🔍 Searching for jobs: {user_query}")
    
    # Search for job URLs using Tavily
    try:
        response = tavily_client.search(
            query=search_query,
            days=days,
            max_results=max_results,
            search_depth='advanced',
            include_domains=include_domains,
            exclude_domains=exclude_domains
        )
        
        job_urls = [result["url"] for result in response["results"]]
        print(f"📋 Found {len(job_urls)} potential job URLs")
        
    except Exception as e:
        print(f"❌ Error searching with Tavily: {e}")
        return JobSearchResult(
            query=user_query,
            total_jobs_found=0,
            jobs=[],
            failed_urls=[]
        )
    
    # Configure Crawl4AI for job extraction
    browser_config = BrowserConfig(
        verbose=True,
        headless=True
    )
    
    # Enhanced extraction instruction for better job data extraction
    extraction_instruction = f"""
    You are a job information extraction specialist. Extract comprehensive job details from this webpage.

    IMPORTANT INSTRUCTIONS:
    1. Look for ALL job-related information on the page
    2. For requirements/skills: Extract both required and preferred qualifications
    3. For salary: Look for any compensation information (salary, hourly rate, equity, bonuses)
    4. For benefits: Extract health insurance, PTO, retirement plans, perks, etc.
    5. For location: Be specific about city/state, remote options, hybrid arrangements
    6. For experience level: Look for years of experience, seniority level mentioned
    7. If multiple jobs are on the page, extract the most prominent/detailed one
    8. Be precise and factual - don't infer information not explicitly stated
    9. For requirements, separate technical skills from soft skills and certifications
    10. Include the original URL in source_url field

    Original search query: {user_query}
    
    Focus on extracting complete, accurate job information that matches or relates to the search query.
    """
    
    run_config = CrawlerRunConfig(
        word_count_threshold=10,
        extraction_strategy=LLMExtractionStrategy(
            llm_config=LLMConfig(
                provider="ollama/qwen3:1.7b", 
                api_token="no-token"
            ),
            schema=JobDetails.model_json_schema(),
            extraction_type="schema",
            instruction=extraction_instruction
        ),
        cache_mode=CacheMode.BYPASS,
        page_timeout=30000,
        wait_for_images=False
    )
    
    extracted_jobs = []
    failed_urls = []
    
    # Extract job details from each URL
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for i, url in enumerate(job_urls, 1):
            print(f"🕷️  Crawling job {i}/{len(job_urls)}: {url}")
            
            try:
                result = await crawler.arun(
                    url=url,
                    config=run_config
                )
                
                if result.extracted_content:
                    # Parse the extracted JSON content
                    try:
                        job_data = json.loads(result.extracted_content)
                        
                        # Handle both single job object and list of jobs
                        if isinstance(job_data, list):
                            # Multiple jobs found on the page
                            for job_item in job_data:
                                if isinstance(job_item, dict):
                                    # Ensure source_url is set
                                    job_item["source_url"] = url
                                    job_details = JobDetails(**job_item)
                                    extracted_jobs.append(job_details)
                                    print(f"✅ Successfully extracted: {job_details.job_title} at {job_details.company_name}")
                        elif isinstance(job_data, dict):
                            # Single job found
                            job_data["source_url"] = url
                            job_details = JobDetails(**job_data)
                            extracted_jobs.append(job_details)
                            print(f"✅ Successfully extracted: {job_details.job_title} at {job_details.company_name}")
                        else:
                            print(f"⚠️  Unexpected data format from {url}: {type(job_data)}")
                            failed_urls.append(url)
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"⚠️  Failed to parse extracted content from {url}: {e}")
                        failed_urls.append(url)
                else:
                    print(f"⚠️  No content extracted from {url}")
                    failed_urls.append(url)
                    
            except Exception as e:
                print(f"❌ Error crawling {url}: {e}")
                failed_urls.append(url)
            
            # Small delay to be respectful to websites
            await asyncio.sleep(1)
    
    result = JobSearchResult(
        query=user_query,
        total_jobs_found=len(extracted_jobs),
        jobs=extracted_jobs,
        failed_urls=failed_urls
    )
    
    print(f"🎯 Extraction complete! Found {len(extracted_jobs)} jobs, {len(failed_urls)} failed URLs")
    return result

def format_job_results(search_result: JobSearchResult, format_type: str = "detailed") -> str:
    """
    Format job search results for display.
    
    Args:
        search_result: JobSearchResult object
        format_type: "summary", "detailed", or "json"
    
    Returns:
        Formatted string representation of the results
    """
    
    if format_type == "json":
        return search_result.model_dump_json(indent=2)
    
    output = []
    output.append(f"🔍 Job Search Results for: '{search_result.query}'")
    output.append(f"📊 Total Jobs Found: {search_result.total_jobs_found}")
    
    if search_result.failed_urls:
        output.append(f"⚠️  Failed URLs: {len(search_result.failed_urls)}")
    
    output.append("=" * 80)
    
    for i, job in enumerate(search_result.jobs, 1):
        output.append(f"\n🏢 JOB #{i}")
        output.append(f"Title: {job.job_title}")
        output.append(f"Company: {job.company_name}")
        
        if job.location:
            output.append(f"Location: {job.location}")
        if job.salary_range:
            output.append(f"Salary: {job.salary_range}")
        if job.job_type:
            output.append(f"Type: {job.job_type}")
        if job.experience_level:
            output.append(f"Experience: {job.experience_level}")
        
        if format_type == "detailed":
            if job.description:
                output.append(f"Description: {job.description}")
            if job.requirements:
                output.append(f"Requirements: {', '.join(job.requirements)}")
            if job.benefits:
                output.append(f"Benefits: {', '.join(job.benefits)}")
            if job.posted_date:
                output.append(f"Posted: {job.posted_date}")
            if job.apply_url:
                output.append(f"Apply: {job.apply_url}")
        
        output.append(f"Source: {job.source_url}")
        output.append("-" * 40)
    
    return "\n".join(output)

# Example usage function
async def main():
    """Example usage of the job search function"""
    
    # Example search queries
    queries = [
        "Python developer remote",
        "DevOps Engineer with Kubernetes experience",
        "Data Scientist machine learning"
    ]
    
    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Testing query: {query}")
        print('=' * 60)
        
        result = await search_and_extract_jobs(
            user_query=query,
            max_results=5,
            days=7
        )
        
        # Display results
        print(format_job_results(result, format_type="summary"))
        
        # Optionally save to file
        with open(f"job_results_{query.replace(' ', '_').lower()}.json", "w") as f:
            f.write(result.model_dump_json(indent=2))
        
        print(f"Results saved to job_results_{query.replace(' ', '_').lower()}.json")

if __name__ == "__main__":
    asyncio.run(main())
