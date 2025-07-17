import asyncio
import os
from typing import List, Optional
from tavily import TavilyClient
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
from datetime import datetime
from config import INCLUDE_DOMAINS, EXCLUDE_DOMAINS, TAVILY_API_KEY, AZURE_OPENAI_API_KEY

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

class JobListings(BaseModel):
    jobs: List[JobDetails] = Field(..., description="List of all job postings found on this page")

class JobSearchResult(BaseModel):
    query: str = Field(..., description="Original search query used")
    total_jobs_found: int = Field(..., description="Total number of jobs successfully extracted")
    jobs: List[JobDetails] = Field(..., description="List of job details extracted")
    failed_urls: List[str] = Field(default_factory=list, description="URLs that failed to be processed")
    enhanced_jobs_count: int = Field(default=0, description="Number of jobs enhanced with apply URL data")

def create_output_directories() -> dict:
    """Create output directories for different types of files."""
    dirs = {
        'search_results': 'output/search_results',
        'raw_crawled': 'output/raw_crawled',
        'enhanced_crawled': 'output/enhanced_crawled'
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def save_job_results(search_result: JobSearchResult, query: str) -> str:
    """Save job search results to JSON file."""
    output_dirs = create_output_directories()
    safe_query = query.replace(' ', '_').replace('/', '_').lower()
    safe_query = ''.join(c for c in safe_query if c.isalnum() or c in ['_', '-'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    filename = f"results_{safe_query}_{timestamp}.json"
    filepath = os.path.join(output_dirs['search_results'], filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(search_result.model_dump_json(indent=2))
    
    return filepath

def get_browser_config() -> BrowserConfig:
    """Get optimized browser configuration."""
    return BrowserConfig(
        headless=True,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        viewport_width=1920,
        viewport_height=1080,
        accept_downloads=False,
        java_script_enabled=True
    )

def get_extraction_config(user_query: str) -> CrawlerRunConfig:
    """Get optimized extraction configuration."""
    extraction_instruction = f"""
    Extract ALL job postings from this webpage as a JSON array. For each job posting, extract:
    - job_title: The exact job title/position name
    - company_name: The hiring company or organization
    - location: Work location, city, state, remote/hybrid status
    - salary_range: Compensation details if mentioned
    - job_type: Employment type (full-time, part-time, contract, etc.)
    - experience_level: Required experience level if stated
    - description: Job summary, responsibilities, role overview
    - requirements: Skills, qualifications, education needed
    - benefits: Company benefits, perks mentioned
    - posted_date: Posting date if available
    - apply_url: Application link or contact method

    Return only jobs relevant to: "{user_query}"
    Output: JSON array of job objects, even if only one job found.
    """
    
    return CrawlerRunConfig(
        word_count_threshold=50,
        extraction_strategy=LLMExtractionStrategy(
            llm_config=LLMConfig(
                provider="azure/thoughtmesh-gpt-4o-mini",
                base_url="https://thoughtmesh-openai.openai.azure.com/",
                api_token=AZURE_OPENAI_API_KEY,
            ),
            schema=JobListings.model_json_schema(),
            extraction_type="schema",
            instruction=extraction_instruction,
            apply_chunking=False,
        ),
        cache_mode=CacheMode.BYPASS,
        page_timeout=30000,
        wait_for_images=False,
        remove_overlay_elements=True,
        simulate_user=True,
        magic=True,
    )

def search_job_urls(user_query: str, max_results: int = 5, days: int = 7) -> List[str]:
    """Search for job URLs using Tavily."""
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    search_query = f"{user_query} jobs hiring careers employment opportunities"
    
    try:
        response = tavily_client.search(
            query=search_query,
            days=days,
            max_results=max_results,
            search_depth='advanced',
            include_domains=INCLUDE_DOMAINS,
            exclude_domains=EXCLUDE_DOMAINS
        )
        return [result["url"] for result in response["results"]]
    except Exception:
        return []

def process_extracted_content(content: str, url: str) -> List[JobDetails]:
    """Process extracted content and return job details."""
    try:
        data = json.loads(content)
        jobs = []
        
        # Handle JobListings format
        if isinstance(data, dict) and "jobs" in data:
            jobs_list = data["jobs"]
        # Handle direct job list format
        elif isinstance(data, list):
            jobs_list = data
        # Handle single job format
        elif isinstance(data, dict) and data.get("job_title"):
            jobs_list = [data]
        else:
            return []
        
        for job_data in jobs_list:
            if isinstance(job_data, dict) and job_data.get("job_title") and job_data.get("company_name"):
                job_data["source_url"] = url
                jobs.append(JobDetails(**job_data))
        
        return jobs
    except Exception:
        return []

def needs_enhancement(job: JobDetails) -> bool:
    """Check if job needs enhancement (has missing important fields)."""
    missing_count = 0
    fields_to_check = [
        job.location, job.salary_range, job.job_type, job.experience_level,
        job.description, job.requirements, job.benefits, job.posted_date
    ]
    
    for field in fields_to_check:
        if not field or (isinstance(field, str) and len(field) < 20) or (isinstance(field, list) and len(field) == 0):
            missing_count += 1
    
    return missing_count > 2 and job.apply_url

async def enhance_job_details(job: JobDetails, crawler: AsyncWebCrawler, user_query: str) -> JobDetails:
    """Enhance job details by crawling apply URL."""
    if not job.apply_url:
        return job
    
    instruction = f"""
    Extract detailed job information for "{job.job_title}" at "{job.company_name}".
    Focus on complete job description, requirements, location, salary, benefits, and posting details.
    Output: Single job object with complete details.
    """
    
    config = CrawlerRunConfig(
        word_count_threshold=50,
        extraction_strategy=LLMExtractionStrategy(
            llm_config=LLMConfig(
                provider="azure/thoughtmesh-gpt-4o-mini",
                base_url="https://thoughtmesh-openai.openai.azure.com/",
                api_token=AZURE_OPENAI_API_KEY,
            ),
            schema=JobDetails.model_json_schema(),
            extraction_type="schema",
            instruction=instruction,
            apply_chunking=False,
        ),
        cache_mode=CacheMode.BYPASS,
        page_timeout=25000,
        wait_for_images=False,
        remove_overlay_elements=True,
        simulate_user=True,
        magic=True,
    )
    
    try:
        result = await crawler.arun(url=job.apply_url, config=config)
        
        if result.extracted_content:
            apply_data = json.loads(result.extracted_content)
            
            if isinstance(apply_data, list) and apply_data:
                apply_data = apply_data[0]
            
            if isinstance(apply_data, dict):
                updated_data = job.model_dump()
                
                for field, new_value in apply_data.items():
                    if field in updated_data and new_value:
                        current_value = updated_data[field]
                        
                        if (current_value is None or
                            (isinstance(current_value, str) and len(current_value.strip()) < 20 and len(str(new_value).strip()) > len(current_value.strip())) or
                            (isinstance(current_value, list) and len(current_value) == 0 and isinstance(new_value, list) and len(new_value) > 0)):
                            updated_data[field] = new_value
                
                updated_data["source_url"] = job.source_url
                updated_data["apply_url"] = job.apply_url
                
                return JobDetails(**updated_data)
    except Exception:
        pass
    
    return job

async def search_and_extract_jobs(
    user_query: str,
    max_results: int = 5,
    days: int = 7,
    max_jobs_to_return: Optional[int] = None,
    enhance_jobs: bool = True
) -> JobSearchResult:
    """
    Search for job postings and extract detailed information.
    
    Args:
        user_query: Job search query
        max_results: Maximum number of URLs to search
        days: Number of days to look back
        max_jobs_to_return: Maximum number of jobs to return
        enhance_jobs: Whether to enhance jobs by crawling apply URLs
    
    Returns:
        JobSearchResult containing extracted job details
    """
    create_output_directories()
    
    # Search for job URLs
    job_urls = search_job_urls(user_query, max_results, days)
    if not job_urls:
        return JobSearchResult(query=user_query, total_jobs_found=0, jobs=[], failed_urls=[])
    
    # Extract jobs from URLs
    browser_config = get_browser_config()
    run_config = get_extraction_config(user_query)
    extracted_jobs = []
    failed_urls = []
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for url in job_urls:
            try:
                result = await crawler.arun(url=url, config=run_config)
                
                if result.extracted_content:
                    jobs = process_extracted_content(result.extracted_content, url)
                    extracted_jobs.extend(jobs)
                else:
                    failed_urls.append(url)
                    
            except Exception:
                failed_urls.append(url)
            
            # Check if we have enough jobs
            if max_jobs_to_return and len(extracted_jobs) >= max_jobs_to_return:
                break
            
            await asyncio.sleep(1)
    
    # Limit jobs to target number
    if max_jobs_to_return:
        extracted_jobs = extracted_jobs[:max_jobs_to_return]
    
    # Enhance jobs if requested
    enhanced_count = 0
    if enhance_jobs and extracted_jobs:
        jobs_to_enhance = [job for job in extracted_jobs if needs_enhancement(job)]
        
        if jobs_to_enhance:
            async with AsyncWebCrawler(config=browser_config) as enhancement_crawler:
                enhanced_jobs = []
                for job in extracted_jobs:
                    if job in jobs_to_enhance:
                        enhanced_job = await enhance_job_details(job, enhancement_crawler, user_query)
                        enhanced_jobs.append(enhanced_job)
                        if enhanced_job != job:
                            enhanced_count += 1
                    else:
                        enhanced_jobs.append(job)
                    await asyncio.sleep(1)
                
                extracted_jobs = enhanced_jobs
    
    result = JobSearchResult(
        query=user_query,
        total_jobs_found=len(extracted_jobs),
        jobs=extracted_jobs,
        failed_urls=failed_urls,
        enhanced_jobs_count=enhanced_count
    )
    
    # Save results
    save_job_results(result, user_query)
    
    return result

async def main():
    """Example usage of the optimized job search function"""
    queries = [
        "Python developer remote",
        # "DevOps Engineer with Kubernetes experience",
        # "Data Scientist machine learning"
    ]
    
    for query in queries:
        result = await search_and_extract_jobs(
            user_query=query,
            max_results=5,
            days=7,
            max_jobs_to_return=3,
            enhance_jobs=True
        )
        
        print(f"Query: {query}")
        print(f"Found {result.total_jobs_found} jobs")
        print(f"Enhanced {result.enhanced_jobs_count} jobs")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
