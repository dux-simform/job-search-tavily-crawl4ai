# Job Search Engine

A comprehensive job search tool that combines Tavily search with Crawl4AI extraction to find and analyze job postings.

## Features

- 🔍 **Smart Search**: Uses Tavily to find recent job postings from multiple job boards
- 🕷️ **AI Extraction**: Leverages Crawl4AI with Azure OpenAI to extract detailed job information
- ✨ **Apply URL Enhancement**: Automatically crawls apply URLs to fill missing job details
- 📊 **Structured Data**: Returns job details in a consistent, structured format
- 🎯 **Customizable**: Configure search domains, time ranges, and result limits
- 💾 **Export Options**: Save results in JSON format for further analysis

## Prerequisites

1. **Tavily API Key**: Get your free API key from [Tavily](https://tavily.com)
2. **Azure OpenAI API Key**: Required for AI-powered job detail extraction
3. **Python 3.10+**: Required for async/await support

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   Create a `.env` file:
   ```
   TAVILY_API_KEY=your_tavily_api_key_here
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
   ```

3. **No additional setup required**: The tool uses Azure OpenAI for AI extraction

## Usage

### Basic Usage

```python
import asyncio
from job_search_engine import search_and_extract_jobs, format_job_results

async def main():
    # Search for jobs with apply URL enhancement (default)
    results = await search_and_extract_jobs(
        user_query="Python developer remote",
        max_results=10,
        days=7,
        enhance_with_apply_urls=True  # NEW: Crawl apply URLs for more details
    )
    
    # Display results
    print(format_job_results(results, format_type="detailed"))
    
    # Access individual jobs
    for job in results.jobs:
        print(f"{job.job_title} at {job.company_name}")
        print(f"Location: {job.location}")
        print(f"Salary: {job.salary_range}")

asyncio.run(main())
```

### Apply URL Enhancement Feature

The tool now automatically crawls job application URLs to fill missing details:

```python
# Enhanced search (slower but more complete)
enhanced_results = await search_and_extract_jobs(
    user_query="Data Scientist machine learning",
    max_results=5,
    enhance_with_apply_urls=True  # Default: enabled
)

print(f"Enhanced {enhanced_results.enhanced_jobs_count} jobs with additional details")

# Fast search (skip apply URL crawling)
quick_results = await search_and_extract_jobs(
    user_query="Data Scientist machine learning", 
    max_results=5,
    enhance_with_apply_urls=False  # Disable for speed
)
```

**When to use enhancement:**
- ✅ Need complete job details
- ✅ Applying to fewer, targeted positions  
- ✅ Data quality > speed

**When to skip enhancement:**
- ✅ Need quick results
- ✅ Broad market research
- ✅ Speed > completeness

### Interactive Examples

Run the enhanced job search example:
```bash
python example_enhanced_usage.py
```

Test the enhancement feature:
```bash
python test_enhancement.py
```

Or run the original examples:
```bash
python example_usage.py
```

### Advanced Configuration

```python
# Custom domain filtering with enhancement
results = await search_and_extract_jobs(
    user_query="DevOps Engineer Kubernetes",
    max_results=15,
    days=14,
    include_domains=["linkedin.com", "indeed.com", "stackoverflow.com"],
    exclude_domains=["sketchy-job-site.com"],
    enhance_with_apply_urls=True  # Enable apply URL enhancement
)

# Different output formats
summary = format_job_results(results, format_type="summary")
detailed = format_job_results(results, format_type="detailed")
json_output = format_job_results(results, format_type="json")

# Check enhancement statistics
print(f"Total jobs: {results.total_jobs_found}")
print(f"Enhanced jobs: {results.enhanced_jobs_count}")
```

## Data Structure

Each job result includes:

```python
{
    "job_title": "Senior Python Developer",
    "company_name": "Tech Corp",
    "location": "San Francisco, CA (Remote friendly)",
    "salary_range": "$120,000 - $160,000",
    "job_type": "Full-time",
    "experience_level": "Senior (5+ years)",
    "description": "We're looking for a senior Python developer...",
    "requirements": [
        "5+ years Python experience",
        "Django/Flask frameworks",
        "AWS cloud experience"
    ],
    "benefits": [
        "Health insurance",
        "401k matching",
        "Remote work options"
    ],
    "posted_date": "2025-01-10",
    "apply_url": "https://company.com/apply",
    "source_url": "https://linkedin.com/jobs/12345"
}
```

## Configuration Options

### Search Parameters

- `user_query`: Your job search terms
- `max_results`: Maximum number of job URLs to process (default: 10)
- `days`: How many days back to search (default: 7)
- `include_domains`: Specific job boards to search (optional)
- `exclude_domains`: Domains to avoid (optional)
- `enhance_with_apply_urls`: Whether to crawl apply URLs for missing details (default: True)

### Default Job Boards

- LinkedIn Jobs
- Indeed
- Glassdoor
- ZipRecruiter
- Y Combinator Jobs
- AngelList
- Stack Overflow Jobs
- Dice
- Monster

## Troubleshooting

### Common Issues

1. **"No TAVILY_API_KEY found"**:
   - Make sure you have a `.env` file with your Tavily API key
   - Verify the key is correct and active

2. **"No AZURE_OPENAI_API_KEY found"**:
   - Ensure you have your Azure OpenAI API key in the `.env` file
   - Check that your Azure OpenAI endpoint is accessible

3. **"No jobs extracted"**:
   - Try a more specific search query
   - Increase the `max_results` parameter
   - Check if the job sites are accessible

4. **Apply URL enhancement is slow**:
   - This is normal - enhancement crawls additional URLs for better data
   - Set `enhance_with_apply_urls=False` for faster results
   - Reduce `max_results` when enhancement is enabled

### Performance Tips

1. **Adjust batch size**: Process fewer URLs at once for stability
2. **Use specific queries**: More targeted searches yield better results
3. **Filter domains**: Focus on job boards that work best for your field
4. **Cache results**: Save results to avoid re-processing the same URLs

## Examples

### Remote Python Jobs
```python
results = await search_and_extract_jobs("Python developer remote work")
```

### Senior DevOps Positions
```python
results = await search_and_extract_jobs(
    "Senior DevOps Engineer AWS Kubernetes", 
    max_results=20
)
```

### Entry-level Data Science
```python
results = await search_and_extract_jobs(
    "entry level data scientist machine learning",
    include_domains=["linkedin.com", "indeed.com"]
)
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License - feel free to use this code for your job search needs.

## Apply URL Enhancement Feature

### How It Works

The apply URL enhancement feature is a two-phase process:

1. **Phase 1 - Initial Extraction**: Extract job details from job listing pages
2. **Phase 2 - Enhancement**: For jobs with missing details and apply URLs, crawl the application pages to fill gaps

### What Gets Enhanced

The system automatically identifies jobs that need enhancement based on missing fields:
- Location information
- Salary/compensation details  
- Job type (full-time, contract, etc.)
- Experience level requirements
- Detailed job descriptions
- Complete requirements lists
- Benefits and perks
- Posting dates

### Enhancement Logic

```python
def has_missing_fields(job: JobDetails) -> bool:
    """Jobs are enhanced if they have 3+ missing important fields"""
    # Checks for missing: location, salary, job_type, experience_level,
    # description, requirements, benefits, posted_date
```

### Performance Considerations

- **Enhancement adds 2-5 seconds per job** that needs enhancement
- **Memory usage**: Minimal impact, processes one apply URL at a time
- **Rate limiting**: Built-in delays respect website policies
- **Failure handling**: Original job data preserved if enhancement fails

### Best Practices

```python
# For job applications (quality over speed)
detailed_jobs = await search_and_extract_jobs(
    user_query="specific role at target companies",
    max_results=5,  # Smaller number for detailed analysis
    enhance_with_apply_urls=True
)

# For market research (speed over completeness)  
market_overview = await search_and_extract_jobs(
    user_query="broad job category",
    max_results=20,  # Larger number for market view
    enhance_with_apply_urls=False
)
```

### Enhancement Statistics

Track enhancement effectiveness:

```python
result = await search_and_extract_jobs("python developer", enhance_with_apply_urls=True)

print(f"Jobs found: {result.total_jobs_found}")
print(f"Jobs enhanced: {result.enhanced_jobs_count}") 
print(f"Enhancement rate: {result.enhanced_jobs_count/result.total_jobs_found:.1%}")
```
