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

# Load environment variables
load_dotenv()

def create_output_directories() -> dict:
    """
    Create output directories for different types of files.
    
    Returns:
        Dictionary with directory paths for different output types
    """
    dirs = {
        'search_results': 'output/search_results',
        'raw_crawled': 'output/raw_crawled',
        'enhanced_crawled': 'output/enhanced_crawled'
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

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
    raw_markdown_files: List[str] = Field(default_factory=list, description="Files containing raw crawled markdown content")
    enhanced_jobs_count: int = Field(default=0, description="Number of jobs enhanced with apply URL data")

def save_job_results_incrementally(search_result: JobSearchResult, query: str, output_dirs: dict, timestamp: str) -> str:
    """
    Save job search results incrementally to avoid data loss.
    
    Args:
        search_result: JobSearchResult object to save
        query: Search query for filename generation
        output_dirs: Dictionary of output directory paths
        timestamp: Consistent timestamp for this search session
        
    Returns:
        The filename where data was saved
    """
    # Generate safe filename
    safe_query = query.replace(' ', '_').replace('/', '_').lower()
    safe_query = ''.join(c for c in safe_query if c.isalnum() or c in ['_', '-'])
    
    # Use consistent filename with session timestamp to overwrite the same file
    filename = f"enhanced_results_{safe_query}_{timestamp}.json"
    filepath = os.path.join(output_dirs['search_results'], filename)
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(search_result.model_dump_json(indent=2))
    
    return filepath

async def search_and_extract_jobs(
    user_query: str,
    max_results: int = 5,
    days: int = 7,
    include_domains: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    max_jobs_to_return: Optional[int] = None
) -> JobSearchResult:
    """
    Search for job postings using Tavily and extract detailed information using Crawl4AI.
    
    Args:
        user_query: User's job search query (e.g., "DevOps Engineer with 5 years experience")
        max_results: Maximum number of job URLs to search for (default: 5)
        days: Number of days to look back for recent postings
        include_domains: List of domains to specifically include in search
        exclude_domains: List of domains to exclude from search
        max_jobs_to_return: Maximum number of jobs to return in final results (None = return all).
                           When specified, crawling will stop early once this target is reached.
    
    Returns:
        JobSearchResult containing extracted job details
        
    Note:
        - Job enhancement is always enabled by default
        - If max_jobs_to_return is specified, URL crawling will terminate early once the target is reached
        - This helps save time and API calls when you only need a specific number of jobs
    """
    
    # Create output directories
    output_dirs = create_output_directories()
    print(f"📁 Output directories created: {list(output_dirs.values())}")
    
    # Get domains from config
    if include_domains is None:
        include_domains = INCLUDE_DOMAINS
    
    if exclude_domains is None:
        exclude_domains = EXCLUDE_DOMAINS
    
    # Initialize Tavily client
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    
    # General search query that finds job-related content across various page types
    search_query = f"{user_query} jobs hiring careers employment opportunities"
    
    print(f"🔍 Searching for jobs: {user_query}")
    
    # Search for job URLs using Tavily
    try:
        # response = tavily_client.search(
        #     query=search_query,
        #     days=days,
        #     max_results=max_results,
        #     search_depth='advanced',
        #     include_domains=include_domains,
        #     exclude_domains=exclude_domains
        # )
        
        # job_urls = [result["url"] for result in response["results"]]
        job_urls = ["https://in.linkedin.com/jobs/view/fullstack-python-developer-remote-work-at-bairesdev-4267988726", "https://in.linkedin.com/jobs/remote-python-jobs-ahmedabad"]
        print(f"📋 Found {len(job_urls)} potential job URLs")
        
    except Exception as e:
        print(f"❌ Error searching with Tavily: {e}")
        return JobSearchResult(
            query=user_query,
            total_jobs_found=0,
            jobs=[],
            failed_urls=[],
            raw_markdown_files=[]
        )
    
    # Configure Crawl4AI with optimized settings for job sites
    browser_config = BrowserConfig(
        verbose=True,
        headless=True,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        viewport_width=1920,
        viewport_height=1080,
        accept_downloads=False,
        java_script_enabled=True
    )
    
    # Multi-job extraction instruction for various page types
    extraction_instruction = f"""
    TASK: Extract ALL job postings from this webpage.

    CONTEXT: You are analyzing a webpage that contains job-related content. This could be:
    - A job board/listing page with multiple job postings
    - A company careers page with several open positions
    - A single job posting page
    - A recruitment page with multiple opportunities
    - Any page mentioning job openings

    EXTRACTION STRATEGY:
    1. SCAN the entire page systematically for ALL job postings
    2. EXTRACT each distinct job opportunity as a separate entry
    3. INCLUDE only jobs that are clearly defined job postings (not just job mentions)
    4. EXTRACT factual information that is explicitly stated for each job
    5. IGNORE navigation, ads, unrelated content, and duplicate listings

    FOR EACH JOB POSTING FOUND, EXTRACT:
    
    job_title: The exact job title/position name
    company_name: The hiring company or organization
    location: Work location, city, state, remote/hybrid status
    salary_range: Compensation details if mentioned
    job_type: Employment type (full-time, part-time, contract, etc.)
    experience_level: Required experience level if stated
    description: Job summary, responsibilities, role overview
    requirements: Skills, qualifications, education needed
    benefits: Company benefits, perks mentioned
    posted_date: Posting date if available
    apply_url: Application link or contact method

    MULTI-JOB EXTRACTION RULES:
    - Extract EVERY distinct job posting on the page
    - Each job should be a complete, separate entry
    - Don't combine information from different jobs
    - If a page has 10 jobs, return all 10 jobs
    - If a page has 1 job, return that 1 job
    - If no clear jobs exist, return empty list
    - Maintain accuracy - only extract what's explicitly stated
    - Don't invent or assume missing information

    QUALITY STANDARDS:
    - Extract verbatim text without interpretation
    - Each job must have at minimum: job_title and company_name
    - Use null for missing information within each job
    - Ensure each job entry is distinct and complete
    - Prioritize jobs relevant to: "{user_query}"

    OUTPUT: JSON array of job objects, even if only one job found
    """
    
    run_config = CrawlerRunConfig(
        word_count_threshold=50,  # Increased to ensure we get substantial content
        extraction_strategy=LLMExtractionStrategy(
            llm_config=LLMConfig(
                provider="azure/thoughtmesh-gpt-4o-mini",
                base_url="https://thoughtmesh-openai.openai.azure.com/",
                api_token=AZURE_OPENAI_API_KEY,
            ),
            schema=JobListings.model_json_schema(),  # Changed to JobListings schema
            extraction_type="schema",
            instruction=extraction_instruction,
            apply_chunking=False,  # Process entire page as one unit
        ),
        cache_mode=CacheMode.BYPASS,
        page_timeout=45000,  # Increased timeout for better page loading
        wait_for_images=False,
        remove_overlay_elements=True,  # Remove popups and overlays
        simulate_user=True,  # Simulate human browsing behavior
        magic=True,  # Enable intelligent content extraction
    )
    
    extracted_jobs = []
    failed_urls = []
    raw_markdown_files = []
    
    # Extract job details from each URL
    target_info = f" (target: {max_jobs_to_return} jobs)" if max_jobs_to_return and max_jobs_to_return > 0 else ""
    print(f"🕷️  Starting to crawl {len(job_urls)} URLs{target_info}")
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for i, url in enumerate(job_urls, 1):
            print(f"🕷️  Crawling job {i}/{len(job_urls)}: {url}")
            
            try:
                result = await crawler.arun(
                    url=url,
                    config=run_config
                )
                
                # Save raw crawled markdown content
                if result.markdown:
                    try:
                        raw_file = save_raw_crawled_markdown(url, result.markdown, user_query, i, output_dirs, "search")
                        raw_markdown_files.append(raw_file)
                        print(f"💾 Saved raw content to: {raw_file}")
                    except Exception as e:
                        print(f"⚠️  Failed to save raw content for {url}: {e}")
                
                if result.extracted_content:
                    # Parse the extracted JSON content
                    try:
                        job_listings_data = json.loads(result.extracted_content)
                        
                        jobs_found_on_page = 0
                        
                        # Handle JobListings format
                        if isinstance(job_listings_data, dict) and "jobs" in job_listings_data:
                            jobs_list = job_listings_data["jobs"]
                            
                            for job_data in jobs_list:
                                if isinstance(job_data, dict):
                                    # Validate essential fields
                                    if job_data.get("job_title") and job_data.get("company_name"):
                                        # Add source_url to each job
                                        job_data["source_url"] = url
                                        job_details = JobDetails(**job_data)
                                        extracted_jobs.append(job_details)
                                        jobs_found_on_page += 1
                                        print(f"✅ Extracted: {job_details.job_title} at {job_details.company_name}")
                                    else:
                                        print(f"⚠️  Skipping incomplete job from {url}")
                                        failed_urls.append(url)
                        
                        # Handle direct job list format (backwards compatibility)
                        elif isinstance(job_listings_data, list):
                            for job_data in job_listings_data:
                                if isinstance(job_data, dict):
                                    if job_data.get("job_title") and job_data.get("company_name"):
                                        job_data["source_url"] = url
                                        job_details = JobDetails(**job_data)
                                        extracted_jobs.append(job_details)
                                        jobs_found_on_page += 1
                                        print(f"✅ Extracted: {job_details.job_title} at {job_details.company_name}")
                                    else:
                                        print(f"⚠️  Skipping incomplete job from {url}")
                                        failed_urls.append(url)
                        
                        # Handle single job format (backwards compatibility)
                        elif isinstance(job_listings_data, dict) and job_listings_data.get("job_title"):
                            if job_listings_data.get("company_name"):
                                job_listings_data["source_url"] = url
                                job_details = JobDetails(**job_listings_data)
                                extracted_jobs.append(job_details)
                                jobs_found_on_page += 1
                                print(f"✅ Extracted single job: {job_details.job_title} at {job_details.company_name}")
                            else:
                                print(f"⚠️  Incomplete single job data from {url}")
                                failed_urls.append(url)
                        
                        else:
                            print(f"⚠️  No valid jobs found in data from {url}")
                            failed_urls.append(url)
                        
                        if jobs_found_on_page > 0:
                            print(f"📋 Found {jobs_found_on_page} jobs on page: {url}")
                            
                            # Show progress towards target if max_jobs_to_return is specified
                            if max_jobs_to_return is not None and max_jobs_to_return > 0:
                                print(f"📊 Progress: {len(extracted_jobs)}/{max_jobs_to_return} jobs collected")
                        
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"⚠️  JSON parsing failed for {url}: {e}")
                        failed_urls.append(url)
                    except Exception as e:
                        print(f"⚠️  Error processing jobs from {url}: {e}")
                        failed_urls.append(url)
                else:
                    print(f"⚠️  No content extracted from {url}")
                    failed_urls.append(url)
                    
            except Exception as e:
                print(f"❌ Error crawling {url}: {e}")
                failed_urls.append(url)
            
            # Check if we have enough jobs based on max_jobs_to_return
            if max_jobs_to_return is not None and max_jobs_to_return > 0:
                if len(extracted_jobs) >= max_jobs_to_return:
                    print(f"🎯 Reached target of {max_jobs_to_return} jobs. Stopping URL crawling early.")
                    break
            
            # Small delay to be respectful to websites
            await asyncio.sleep(1)
    
    # Limit the jobs to the target number BEFORE enhancement
    final_jobs_to_process = extracted_jobs
    if max_jobs_to_return is not None and max_jobs_to_return > 0:
        final_jobs_to_process = extracted_jobs[:max_jobs_to_return]
        if len(extracted_jobs) > max_jobs_to_return:
            print(f"📊 Selecting first {max_jobs_to_return} jobs for processing (out of {len(extracted_jobs)} found)")
    
    # Second pass: Enhance jobs with missing details by crawling apply URLs (always enabled)
    enhanced_jobs = []
    if final_jobs_to_process:
        print(f"🔍 Checking for jobs that need enhancement...")
        jobs_to_enhance = [job for job in final_jobs_to_process if job.apply_url and has_missing_fields(job)]
        enhanced_jobs.extend([job for job in final_jobs_to_process if job not in jobs_to_enhance])

        if jobs_to_enhance:
            # Initialize timestamp for this enhancement session
            enhancement_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            print(f"🚀 Found {len(jobs_to_enhance)} jobs with apply URLs that need enhancement")
            
            async with AsyncWebCrawler(config=browser_config) as enhancement_crawler:
                for i, job in enumerate(jobs_to_enhance, 1):
                    print(f"🔗 Enhancing job {i}/{len(jobs_to_enhance)}: {job.job_title} at {job.company_name}")
                    
                    enhanced_job = await crawl_apply_url_for_details(job, enhancement_crawler, user_query, output_dirs, i)
                    enhanced_jobs.append(enhanced_job)
                    
                    # Save incremental results after enhancement
                    try:
                        # Only save the enhanced jobs, not all jobs
                        enhanced_result = JobSearchResult(
                            query=user_query,
                            total_jobs_found=len(enhanced_jobs),
                            jobs=enhanced_jobs,
                            failed_urls=[],
                            raw_markdown_files=[],
                            enhanced_jobs_count=len(enhanced_jobs)
                        )
                        incremental_file = save_job_results_incrementally(enhanced_result, user_query, output_dirs, enhancement_timestamp)
                        print(f"💾 Saved enhanced results to: {incremental_file}")
                    except Exception as e:
                        print(f"⚠️  Failed to save enhanced incremental results: {e}")
                    
                    # Small delay between enhancement requests
                    await asyncio.sleep(2)
            
            # Replace original jobs with enhanced versions in the final_jobs_to_process set
            enhanced_job_map = {job.apply_url: job for job in enhanced_jobs}
            final_jobs = []
            
            for job in final_jobs_to_process:
                if job.apply_url in enhanced_job_map:
                    final_jobs.append(enhanced_job_map[job.apply_url])
                else:
                    final_jobs.append(job)
            
            final_jobs_to_process = final_jobs
            print(f"✨ Enhancement complete! Enhanced {len(enhanced_jobs)} jobs with additional details")
        else:
            print(f"ℹ️  No jobs found that need enhancement or have apply URLs")
    else:
        print(f"ℹ️  No jobs extracted to enhance")
    
    result = JobSearchResult(
        query=user_query,
        total_jobs_found=len(final_jobs_to_process),
        jobs=final_jobs_to_process,
        failed_urls=failed_urls,
        raw_markdown_files=raw_markdown_files,
        enhanced_jobs_count=len(enhanced_jobs)
    )
    
    print(f"🎯 Extraction complete! Found {len(final_jobs_to_process)} jobs, {len(failed_urls)} failed URLs")
    if enhanced_jobs:
        print(f"✨ Enhanced {len(enhanced_jobs)} jobs with additional details from apply URLs")
    print(f"💾 Saved {len(raw_markdown_files)} raw markdown files")
    return result

def save_raw_crawled_markdown(url: str, markdown_content: str, query: str, index: int, output_dirs: dict, crawl_type: str = "search") -> str:
    """
    Save raw crawled markdown content from crawler result.
    
    Args:
        url: The URL that was crawled
        markdown_content: The raw markdown content from result.markdown
        query: The search query for filename generation
        index: Index number for the crawled page
        output_dirs: Dictionary of output directory paths
        crawl_type: Type of crawl ("search" or "enhanced") to determine subdirectory
    
    Returns:
        The filename where the raw content was saved
    """
    # Generate safe filename
    safe_query = query.replace(' ', '_').replace('/', '_').lower()
    safe_query = ''.join(c for c in safe_query if c.isalnum() or c in ['_', '-'])
    
    # Extract domain from URL for filename
    from urllib.parse import urlparse
    domain = urlparse(url).netloc.replace('www.', '').replace('.', '_')
    
    # Choose directory based on crawl type
    if crawl_type == "enhanced":
        base_dir = output_dirs['enhanced_crawled']
        filename = f"enhanced_crawled_{safe_query}_{index:02d}_{domain}.md"
    else:
        base_dir = output_dirs['raw_crawled']
        filename = f"raw_crawled_{safe_query}_{index:02d}_{domain}.md"
    
    filepath = os.path.join(base_dir, filename)
    
    content = []
    content.append("# Raw Crawled Content")
    content.append(f"**URL:** {url}")
    content.append(f"**Query:** {query}")
    content.append(f"**Crawl Type:** {crawl_type.title()}")
    content.append(f"**Crawled Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content.append(f"**Page Index:** {index}")
    content.append("\n" + "="*80 + "\n")
    content.append(markdown_content)
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))
    
    return filepath

def has_missing_fields(job: JobDetails) -> bool:
    """
    Check if a job has missing important fields that could be filled by crawling apply URL.
    
    Args:
        job: JobDetails object to check
        
    Returns:
        True if job has missing fields, False otherwise
    """
    missing_fields = []
    
    # Check for missing important fields
    if not job.location:
        missing_fields.append("location")
    if not job.salary_range:
        missing_fields.append("salary_range")
    if not job.job_type:
        missing_fields.append("job_type")
    if not job.experience_level:
        missing_fields.append("experience_level")
    if not job.description or len(job.description) < 50:
        missing_fields.append("description")
    if not job.requirements or len(job.requirements) == 0:
        missing_fields.append("requirements")
    if not job.benefits or len(job.benefits) == 0:
        missing_fields.append("benefits")
    if not job.posted_date:
        missing_fields.append("posted_date")
    
    return len(missing_fields) > 2  # If more than 2 important fields are missing

async def crawl_apply_url_for_details(
    job: JobDetails, 
    crawler: AsyncWebCrawler,
    user_query: str,
    output_dirs: dict,
    index: int
) -> JobDetails:
    """
    Crawl the apply URL to extract missing job details.
    
    Args:
        job: JobDetails object with apply_url to crawl
        crawler: AsyncWebCrawler instance
        user_query: Original search query for context
        output_dirs: Dictionary of output directory paths
        index: Index number for the enhanced crawl
        
    Returns:
        Updated JobDetails object with filled missing fields
    """
    if not job.apply_url:
        return job
    
    print(f"🔗 Crawling apply URL for additional details: {job.apply_url}")
    
    # Create extraction instruction focused on filling missing details
    missing_fields_instruction = f"""
    TASK: Extract detailed job information from this job application page.

    CONTEXT: This is a job application or job detail page for the position "{job.job_title}" at "{job.company_name}".
    
    FOCUS: Extract comprehensive job details, paying special attention to:
    - Complete job description and responsibilities
    - Detailed requirements, skills, and qualifications
    - Location information (city, state, remote/hybrid options)
    - Salary, compensation, or pay range
    - Employment type (full-time, part-time, contract, etc.)
    - Experience level required
    - Company benefits and perks
    - Application deadline or posting date
    - Any application instructions or contact information

    EXTRACTION RULES:
    - Extract only factual information explicitly stated
    - Be comprehensive but accurate
    - Don't invent or assume missing information
    - Focus on job-specific details rather than company general info
    - Extract verbatim text without interpretation

    OUTPUT: Single job object with complete details (not a list, just one job object)
    """
    
    run_config = CrawlerRunConfig(
        word_count_threshold=50,
        extraction_strategy=LLMExtractionStrategy(
            llm_config=LLMConfig(
                provider="azure/thoughtmesh-gpt-4o-mini",
                base_url="https://thoughtmesh-openai.openai.azure.com/",
                api_token=AZURE_OPENAI_API_KEY,
            ),
            schema=JobDetails.model_json_schema(),
            extraction_type="schema",
            instruction=missing_fields_instruction,
            apply_chunking=False,
        ),
        cache_mode=CacheMode.BYPASS,
        page_timeout=30000,
        wait_for_images=False,
        remove_overlay_elements=True,
        simulate_user=True,
        magic=True,
    )
    
    try:
        result = await crawler.arun(
            url=job.apply_url,
            config=run_config
        )
        
        # Save raw crawled markdown content for enhanced URLs
        if result.markdown:
            try:
                enhanced_raw_file = save_raw_crawled_markdown(
                    job.apply_url, result.markdown, user_query, index, output_dirs, "enhanced"
                )
                print(f"💾 Saved enhanced raw content to: {enhanced_raw_file}")
            except Exception as e:
                print(f"⚠️  Failed to save enhanced raw content for {job.apply_url}: {e}")
        
        if result.extracted_content:
            try:
                apply_job_data = json.loads(result.extracted_content)
                print(f"🔍 Apply URL returned data type: {type(apply_job_data)}")
                
                # Handle both list and dict formats
                job_data_to_merge = None
                if isinstance(apply_job_data, list) and len(apply_job_data) > 0:
                    # If it's a list, take the first job
                    job_data_to_merge = apply_job_data[0]
                    print("📋 Using first job from list format")
                elif isinstance(apply_job_data, dict):
                    # If it's a dict, use it directly
                    job_data_to_merge = apply_job_data
                    print("📋 Using dict format")
                
                if job_data_to_merge and isinstance(job_data_to_merge, dict):
                    # Merge the new data with existing job data, prioritizing existing non-null values
                    updated_job_data = job.model_dump()
                    fields_updated = []
                    
                    # Only update fields that are currently missing or incomplete
                    for field, new_value in job_data_to_merge.items():
                        if field in updated_job_data and new_value:
                            current_value = updated_job_data[field]
                            
                            # Update if current field is None or empty
                            if current_value is None:
                                updated_job_data[field] = new_value
                                fields_updated.append(field)
                            # For string fields, update if current is very short and new is longer
                            elif isinstance(current_value, str) and isinstance(new_value, str):
                                if len(current_value.strip()) < 20 and len(new_value.strip()) > len(current_value.strip()):
                                    updated_job_data[field] = new_value
                                    fields_updated.append(field)
                            # For list fields, update if current is empty and new has content
                            elif isinstance(current_value, list) and isinstance(new_value, list):
                                if len(current_value) == 0 and len(new_value) > 0:
                                    updated_job_data[field] = new_value
                                    fields_updated.append(field)
                    
                    # Ensure source_url and apply_url are preserved
                    updated_job_data["source_url"] = job.source_url
                    updated_job_data["apply_url"] = job.apply_url
                    
                    updated_job = JobDetails(**updated_job_data)
                    print(f"✅ Enhanced {len(fields_updated)} fields: {', '.join(fields_updated)}")
                    return updated_job
                else:
                    print(f"⚠️  No valid job data found in apply URL response")
                
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️  Failed to parse apply URL content for {job.job_title}: {e}")
            except Exception as e:
                print(f"⚠️  Error processing apply URL for {job.job_title}: {e}")
        else:
            print(f"⚠️  No content extracted from apply URL: {job.apply_url}")
            
    except Exception as e:
        print(f"❌ Error crawling apply URL {job.apply_url}: {e}")
    
    return job  # Return original job if enhancement failed

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
        
        # Run search with enhancement enabled by default
        print("🚀 Running job search with enhancement...")
        result = await search_and_extract_jobs(
            user_query=query,
            max_results=5,  # Search 5 URLs
            days=7,
            max_jobs_to_return=None  # Return all jobs found
        )
        
        # Display results
        print(f"🎯 Found {result.total_jobs_found} jobs")
        
        print("Results automatically saved to output/ directory:")
        print(f"  - Search results: output/search_results/")
        if result.raw_markdown_files:
            print(f"  - Raw content: {len(result.raw_markdown_files)} files in output/raw_crawled/ and output/enhanced_crawled/")
        if result.enhanced_jobs_count > 0:
            print(f"  - Enhanced jobs: {result.enhanced_jobs_count}")

if __name__ == "__main__":
    asyncio.run(main())
