# Job Search Engine

A comprehensive job search tool that combines Tavily search with Crawl4AI extraction to find and analyze job postings.

## Features

- 🔍 **Smart Search**: Uses Tavily to find recent job postings from multiple job boards
- 🕷️ **AI Extraction**: Leverages Crawl4AI with local Ollama models to extract detailed job information
- 📊 **Structured Data**: Returns job details in a consistent, structured format
- 🎯 **Customizable**: Configure search domains, time ranges, and result limits
- 💾 **Export Options**: Save results in JSON format for further analysis

## Prerequisites

1. **Tavily API Key**: Get your free API key from [Tavily](https://tavily.com)
2. **Ollama**: Install and run Ollama with the qwen3:1.7b model
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
   ```

3. **Install and start Ollama**:
   ```bash
   # Install Ollama (if not already installed)
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Pull the required model
   ollama pull qwen3:1.7b
   
   # Start Ollama service
   ollama serve
   ```

## Usage

### Basic Usage

```python
import asyncio
from job_search_engine import search_and_extract_jobs, format_job_results

async def main():
    # Search for jobs
    results = await search_and_extract_jobs(
        user_query="Python developer remote",
        max_results=10,
        days=7
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

### Interactive Example

Run the interactive example:
```bash
python example_usage.py
```

Or run predefined examples:
```bash
python example_usage.py --examples
```

### Advanced Configuration

```python
# Custom domain filtering
results = await search_and_extract_jobs(
    user_query="DevOps Engineer Kubernetes",
    max_results=15,
    days=14,
    include_domains=["linkedin.com", "indeed.com", "stackoverflow.com"],
    exclude_domains=["sketchy-job-site.com"]
)

# Different output formats
summary = format_job_results(results, format_type="summary")
detailed = format_job_results(results, format_type="detailed")
json_output = format_job_results(results, format_type="json")
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

2. **"Ollama connection failed"**:
   - Ensure Ollama is running: `ollama serve`
   - Check if the model is available: `ollama list`
   - Pull the model if needed: `ollama pull qwen2:1.5b`

3. **"No jobs extracted"**:
   - Try a more specific search query
   - Increase the `max_results` parameter
   - Check if the job sites are accessible

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
