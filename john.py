from g4f import Provider, models
from langchain.llms.base import LLM
from langchain_g4f import G4FLLM
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
import os
import json
import requests
import asyncio
import nest_asyncio
from newspaper import Article
from trafilatura import fetch_url, extract
import xml.etree.ElementTree as ET
import feedparser
import fitz
import io
from PIL import Image

nest_asyncio.apply()
# Set up the LLM model
llm: LLM = G4FLLM(
    model=models.gpt_35_turbo,
    provider=Provider.FreeGpt,
)


async def extract_square_image_from_pdf(pdf_path, min_resolution):
    doc = fitz.open(pdf_path)
    square_image = None
    min_diff = float("inf")
    image_count = 0

    for i in range(len(doc)):
        for img in doc.get_page_images(i):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]

            # Open the image using PIL
            image = Image.open(io.BytesIO(image_data))

            # Skip if the image is smaller than the minimum resolution
            if image.width < min_resolution or image.height < min_resolution:
                continue

            # If the image is perfectly square, return it immediately
            if image.width == image.height:
                return image

            # Calculate the difference between width and height
            diff = abs(image.width - image.height)

            # If this image is more square than the previous most square image,
            # update min_diff and square_image
            if diff < min_diff:
                min_diff = diff
                square_image = image

            # Increment the count of checked images
            image_count += 1

            # If we've checked 10 images, return the most square one found so far
            if image_count >= 10:
                return square_image

    return square_image


async def createPost(
    prompt: PromptTemplate,
    output_parser: StructuredOutputParser,
    text: str,
    image_url: str = None,
):
    _input = prompt.format_prompt(information=text)
    # print(_input.to_string())
    print(len(_input.to_string()))

    output = llm(_input.to_string())
    print(output)

    parsed_output = output_parser.parse(output)

    # Access the "title" and "description"
    title = parsed_output.get("title")
    description = parsed_output.get("description")
    full_explanation = parsed_output.get("full_explanation")

    if image_url is None:
        search_query = parsed_output.get("search_query")
        # fetch image from google
        image_url = await fetch_google_image(search_query)
        print("Search Query:", search_query)

    print("Title:", title)
    print("Description:", description)
    print("Image Url:", image_url)
    # print("Full Explanation:", full_explanation)

    # Define the URL of your server
    url = "https://nodejs-serverless-function-express-snowy-eight.vercel.app"  # Replace with the actual URL

    # Define the request payload (body)
    payload = {
        "title": title,
        "description": description,
        "imageUrl": image_url,
        "full_explanation": full_explanation,
    }

    # Make a POST request to the server
    response = await asyncio.to_thread(
        requests.post, url + "/cells/create", json=payload
    )
    return response


async def fetch_google_image(query: str):
    print(query)
    # Set up the endpoint
    endpoint = "https://www.googleapis.com/customsearch/v1"

    # Set up the request parameters
    params = {
        "key": "AIzaSyCJyppT0IrWrsLf2V7mQDvG6McfnWuer-s",  # Your API key
        "cx": "42777fbc0b26042dc",  # Your Search Engine ID
        "q": query,  # The search query
        "searchType": "image",  # To search for images
        "num": 10,  # The number of images to return
    }

    # Send the GET request
    response = await asyncio.to_thread(requests.get, endpoint, params=params)

    # Get the JSON response
    json_response = response.json()

    # Initialize min_diff with a large value
    min_diff = float("inf")
    square_image_link = None

    # Iterate over the image results
    for image_result in json_response["items"]:
        width = image_result["image"]["width"]
        height = image_result["image"]["height"]

        # Calculate the difference between width and height
        diff = abs(width - height)

        # If this image is more square than the previous most square image,
        # update min_diff and square_image_link
        if diff < min_diff:
            min_diff = diff
            square_image_link = image_result["link"]

    return square_image_link


class ArxivPostCreator:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, "arxiv_used_dois.txt")
        self.processed_dois = set()
        response_schemas = [
            ResponseSchema(
                name="title",
                description="Generate a precise title that captures the essence of the information in 1-4 keywords. Make it very short and understadable by broad audience ",
            ),
            ResponseSchema(
                name="description",
                description="Elaborate on the title with a short one-sentence summary that provides new insights.",
            ),
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(
            response_schemas
        )

        format_instructions = self.output_parser.get_format_instructions()
        self.prompt = PromptTemplate(
            template="Summary should be in an unbiased manner. Create an understandable title and short description of the information\n{format_instructions}\n{information}",
            # include - to keep it understandable and not use complex words, and write as neutral as possible
            input_variables=["information"],
            partial_variables={"format_instructions": format_instructions},
        )

    async def load_processed_dois(self):
        if not os.path.exists(self.output_file):
            # Create the file if it doesn't exist
            with open(self.output_file, "w"):
                pass
        with open(self.output_file, "r") as f:
            lines = f.read().splitlines()
            self.processed_dois.update(map(str, lines))

    async def save_processed_dois(self):
        with open(self.output_file, "w") as f:
            f.write("\n".join(map(str, self.processed_dois)))

    async def fetch_arxiv_data(self, category, search_query, start=0, max_results=10):
        url = f"http://export.arxiv.org/api/query?search_query={category}:{search_query}&start={start}&max_results={max_results}"

        response = requests.get(url)

        if response.status_code == 200:
            feed = feedparser.parse(response.content)

            processed = []

            for entry in feed.entries:
                doi = entry.get("id", None)
                if doi and doi not in self.processed_dois:
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")

                    # Get the link to the PDF
                    pdf_link = ""
                    for link in entry.get("links", []):
                        if link.get("type") == "application/pdf":
                            pdf_link = link.get("href")
                            break

                    # Store the information in a dictionary
                    post_info = {
                        "doi": doi,
                        "title": title,
                        "summary": summary,
                        "pdf_link": pdf_link,
                    }
                    print(post_info)

                    processed.append(post_info)
            return processed

    async def process_arxiv_post(self, post_info):
        doi = post_info["doi"]
        title = post_info["title"]
        summary = post_info["summary"]
        pdf_link = post_info["pdf_link"]
        text = f"{title}\n{summary}"

        image_url = await extract_square_image_from_pdf(pdf_link, 200)
        response = await createPost(self.prompt, self.output_parser, text, image_url)
        if response.status_code == 200:
            self.processed_dois.add(doi)  # Mark the post as processed
            await self.save_processed_dois()  # Save the updated list of processed IDs
            print("Success:", response.json())
        else:
            print(f"{response.status_code} Server error:", response.json())

    async def main(self):
        await self.load_processed_dois()
        posts = await self.fetch_arxiv_data("all", "machine learning", 0, 10)

        # Create tasks to fetch details for each story concurrently
        tasks = [self.process_arxiv_post(post_info) for post_info in posts]
        await asyncio.gather(*tasks)


class HackerNewsPostCreator:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.output_file = os.path.join(output_dir, "hackernews_used_ids.txt")
        self.processed_ids = set()
        self.semaphore = asyncio.Semaphore(10)
        response_schemas = [
            ResponseSchema(
                name="full_explanation",
                description="Generate a detailed and long full explanation/summarization of the information",
            ),
            ResponseSchema(
                name="title",
                description="Generate a precise title that captures the essence of the information in 1-3 keywords. Make it understadable by broad audience and very short",
            ),
            ResponseSchema(
                name="description",
                description="Elaborate on the title with a short one-sentence summary that provides new insights.",
            ),
            ResponseSchema(
                name="search_query",
                description="Generate a search query for google images to find the most representative picture of the exact information.",
            ),
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(
            response_schemas
        )

        format_instructions = self.output_parser.get_format_instructions()
        self.prompt = PromptTemplate(
            template="Summarize the information in an unbiased manner. Write as if you're the author sharing this information.\n{format_instructions}\n{information}",
            # include - to keep it understandable and not use complex words, and write as neutral as possible
            input_variables=["information"],
            partial_variables={"format_instructions": format_instructions},
        )

    async def load_processed_ids(self):
        if not os.path.exists(self.output_file):
            # Create the file if it doesn't exist
            with open(self.output_file, "w") as f:
                pass
        with open(self.output_file, "r") as f:
            lines = f.read().splitlines()
            self.processed_ids.update(
                map(str, lines)
            )  # Convert loaded values to strings

    async def save_processed_ids(self):
        with open(self.output_file, "w") as f:
            f.write("\n".join(map(str, self.processed_ids)))

    async def parseSite(self, story_id: str):
        if story_id in self.processed_ids:
            print("exists")
            return None  # Skip processing if the post has already been processed

        item_response = await asyncio.to_thread(
            requests.get,
            f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json?print=pretty",
        )
        item_response.raise_for_status()
        story_details = json.loads(item_response.text)
        # print(story_details)
        title = story_details.get("title")
        url = story_details.get("url")
        if not url or not title:
            return None

        # grab a HTML file to extract data from
        downloaded = fetch_url(url)

        # output main content and comments as plain text
        result = extract(downloaded, include_comments=False)

        main_text = result

        # Limit main_text to a maximum of 10,000 characters
        main_text = main_text[:10000]

        if len(main_text) < 200:
            return None  # Skip processing if the content is too short
        else:
            return f"\nText:\n{title}\n{main_text}"

    async def process_post(
        self,
        story_id: str,
    ):
        async with self.semaphore:
            try:
                text = await self.parseSite(story_id)
                if text is None:
                    return
                # Check the response
                response = await createPost(self.prompt, self.output_parser, text)
                if response.status_code == 200:
                    self.processed_ids.add(story_id)  # Mark the post as processed
                    await self.save_processed_ids()  # Save the updated list of processed IDs
                    print("Success:", response.json())
                else:
                    print(f"{response.status_code} Server error:", response.json())
            except Exception as e:
                print(f"Error processing post {story_id}: {str(e)}")

    async def main(self):
        await self.load_processed_ids()

        response = await asyncio.to_thread(
            requests.get,
            "https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty",
        )

        response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
        response2 = await asyncio.to_thread(
            requests.get,
            "https://hacker-news.firebaseio.com/v0/beststories.json?print=pretty",
        )
        response2.raise_for_status()
        # merge and remove duplicates
        # top and best stories combined
        story_ids = list(set(eval(response.text) + eval(response2.text)))
        # print(story_ids)

        # Create tasks to fetch details for each story concurrently
        tasks = [self.process_post(str(story_id)) for story_id in story_ids]

        # Run the tasks concurrently
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    # You can now work with the categorized_posts list as needed
    # It contains dictionaries with "Title" and "Category" keys for each post

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    # hacker_news_post_creator = HackerNewsPostCreator(output_dir)
    arxiv_post_creator = ArxivPostCreator(output_dir)

    # asyncio.run(hacker_news_post_creator.main())
    asyncio.run(arxiv_post_creator.main())
