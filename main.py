from g4f import Provider, models
from langchain.llms.base import LLM
from langchain_g4f import G4FLLM
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
import os
import json
import asyncio
import nest_asyncio
from newspaper import Article
from trafilatura import fetch_url, extract
import feedparser
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from newspaper import Article
import re
from typing import List  # Import the List type hint
import base64
import time
import aiohttp
import requests

nest_asyncio.apply()

# Set up the LLM model
llm: LLM = G4FLLM(
    model=models.gpt_35_turbo,
    provider=Provider.GeekGpt,
)


async def upload_image(extracted_image_url):
    try:
        # Convert the image to base64
        async with aiohttp.ClientSession() as session:
            async with session.get(extracted_image_url) as response:
                image_content = await response.read()
                image_base64 = base64.b64encode(image_content).decode("utf-8")

        if image_base64:
            url = "https://api.imgbb.com/1/upload?key=c16460ec60fe42c07eb757018ea9e5dd"
            payload = {"image": image_base64}

            async with aiohttp.ClientSession() as session:
                async with session.post(url, data=payload) as res:
                    if res.status != 200:
                        raise Exception(
                            "Failed to upload image. Status code: " + str(res.status)
                        )

                    data = await res.json()

                    if data["success"]:
                        return data["data"]["url"]
                    else:
                        print("Error uploading image:", data)
                        return None
        else:
            return None
    except:
        return None


def extract_youtube_video_id(url):
    # Extract the YouTube video ID from the URL
    video_id = None

    if "youtu.be" in url:
        # For short youtu.be URLs
        video_id = url.split("/")[-1].split("?")[0]
    else:
        # For standard YouTube URLs
        match = re.search(r"([A-Za-z0-9_-]{11})", url)
        if match:
            video_id = match.group(1)

    return video_id


async def get_youtube_transcript(video_url):
    # extract youtube transcript from the video
    video_id = extract_youtube_video_id(video_url)
    print(video_id)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    formatter = TextFormatter()
    plain_text = formatter.format_transcript(transcript)
    return plain_text


def login(username, password):
    url = "https://gateapi.vercel.app/users/auth"
    payload = {
        "action": "login",
        "data": {"identifier": username, "password": password},
    }

    response = requests.post(url, json=payload)
    data = response.json()
    if response.status_code == 200:
        token = data["token"]
        with open("token.txt", "w") as f:
            f.write(token)
    else:
        print(f"{response.status_code} Server error:", data)


async def extract_best_image_from_page(page_url):
    try:
        article = Article(page_url)
        article.download()
        article.parse()

        # Get the main image URL
        best_image_url = article.top_image
        if best_image_url:
            return best_image_url

    except Exception as e:
        print(f"Error extracting the best image from the page: {str(e)}")

    # Return None if no suitable image is found
    return None


async def createPost(
    prompt: PromptTemplate,
    output_parser: StructuredOutputParser,
    text: str,
    token: str,
    links: List[str] = None,
    image_url: str = None,
):
    # later rotate LLMs to always use working one
    try:
        _input = prompt.format_prompt(information=text)
        # print(_input.to_string())
        print(len(_input.to_string()))

        output = llm(_input.to_string())
        print(output)

        parsed_output = output_parser.parse(output)
    except Exception as e:
        err = f"LLM Error: {e}"

        return (err, None, None)

    # Access the "title" and "description"
    title = parsed_output.get("title")
    description = parsed_output.get("description")
    full_explanation = parsed_output.get("full_explanation")

    if image_url is None:
        search_query = parsed_output.get("search_query")
        # fetch image from google
        google_image_url = await fetch_google_image(search_query)
        image_url = await upload_image(google_image_url)
        print("Search Query:", search_query)

    print("Title:", title)
    print("Description:", description)
    print("Image Url:", image_url)
    # print("Full Explanation:", full_explanation)

    # Define the URL of your server
    url = "https://gateapi.vercel.app"  # Replace with the actual URL

    # Define the request payload (body)
    payload = {
        "title": title,
        "description": description,
        "imageUrl": image_url,
        "full_explanation": full_explanation,
        "links": links,
    }

    headers = {"Authorization": token}

    async with aiohttp.ClientSession() as session:
        async with session.post(
            url + "/cells/create", headers=headers, json=payload
        ) as response:
            data = await response.json()
            return (None, response, data)


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

    # Initialize min_diff with a large value
    min_diff = float("inf")
    square_image_link = None

    async with aiohttp.ClientSession() as session:
        async with session.get(endpoint, params=params) as response:
            json_response = await response.json()

            # Iterate over the image results
            for image_result in json_response["items"]:
                width = image_result["image"]["width"]
                height = image_result["image"]["height"]
                diff = abs(width - height)

                if diff < min_diff:
                    min_diff = diff
                    square_image_link = image_result["link"]

    return square_image_link


class HackerNewsPostCreator:
    def __init__(self, output_dir="output", semaphores=5):
        self.output_dir = output_dir
        self.output_file_ids = os.path.join(output_dir, "hackernews_used_ids.txt")
        self.output_file_urls = os.path.join(output_dir, "hackernews_used_urls.txt")
        self.processed_ids = set()
        self.processed_urls = set()
        with open("token.txt", "r") as f:
            self.token = f.read()
        self.semaphore = asyncio.Semaphore(semaphores)
        response_schemas = [
            ResponseSchema(
                name="full_explanation",
                description="Generate a detailed and full summarization of the information. This should be long and extensive. Structure it in paragraphs and include all information possible. ",
            ),
            ResponseSchema(
                name="title",
                description="Generate a precise title that captures the essence of the information in keywords. Make it understadable by broad audience and very short. Can be as short as one word.",
            ),
            ResponseSchema(
                name="description",
                description="Elaborate on the title with a short, one-sentence description.",
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
            template="Summarize the information in an unbiased manner.\n{format_instructions}\n{information}",
            # include - to keep it understandable and not use complex words, and write as neutral as possible
            input_variables=["information"],
            partial_variables={"format_instructions": format_instructions},
        )

    async def load_processed_ids(self):
        if os.path.exists(self.output_file_ids):
            with open(self.output_file_ids, "r") as f:
                self.processed_ids = set(f.read().splitlines())
        if os.path.exists(self.output_file_urls):
            with open(self.output_file_urls, "r") as f:
                self.processed_urls = set(f.read().splitlines())

    async def save_processed_ids(self):
        with open(self.output_file_ids, "w") as f:
            f.write("\n".join(self.processed_ids))
        with open(self.output_file_urls, "w") as f:
            f.write("\n".join(self.processed_urls))

    async def parseSite(self, story_id: str) -> (str, str):
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json?print=pretty"
            ) as response:
                response.raise_for_status()
                story_details = await response.json()
        # print(story_details)
        title = story_details.get("title")
        url = story_details.get("url")
        print(url, title)
        if not url or not title:
            print("hacker news has no title or url")
            return None, None

        if url in self.processed_urls:
            print("url exists")
            return None, None

        if story_id in self.processed_ids:
            print("story_id exists")
            return None, None

        text_contents = None

        # if url is youtube get transcript
        if "youtube.com" in url or "youtu.be" in url:
            # extract it
            text_contents = await get_youtube_transcript(url)
        else:
            # grab a HTML file to extract data fro
            downloaded = fetch_url(url)

            # output main content and comments as plain text
            text_contents = extract(downloaded, include_comments=False)

        if not text_contents:
            print("no text")
            return None, None

        if len(text_contents) < 200:
            print("too short")
            return None, None  # Skip processing if the content is too short

        # Limit text_contents to a maximum of 12,000 characters
        text_contents = text_contents[:12000]

        return (f"\nText:\n{title}\n{text_contents}", url)

    async def process_post(
        self,
        story_id: str,
    ):
        async with self.semaphore:
            try:
                text, url = await self.parseSite(story_id)

                if text is None or url is None:
                    return

                image_url = None

                extracted_image_url = await extract_best_image_from_page(url)
                try:
                    image_url = await upload_image(extracted_image_url)
                except:
                    image_url = None
                links = [url]

                # Check the response
                err, response, data = await createPost(
                    self.prompt, self.output_parser, text, self.token, links, image_url
                )
                if err:
                    return
                if response.status == 200:
                    self.processed_ids.add(story_id)
                    self.processed_urls.add(url)
                    await self.save_processed_ids()
                    print("Success:", data)
                else:
                    print(f"{response.status} Server error:", data)
            except Exception as e:
                raise Exception(f"Error processing post {story_id}: {str(e)}")

    async def run(self):
        await self.load_processed_ids()

        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty"
            ) as response:
                response.raise_for_status()
                story_ids1 = await response.json()

            async with session.get(
                "https://hacker-news.firebaseio.com/v0/beststories.json?print=pretty"
            ) as response:
                response.raise_for_status()
                story_ids2 = await response.json()
        # merge and remove duplicates
        # top and best stories combined
        story_ids = list(set(story_ids1 + story_ids2))

        # Filter out story IDs that have already been processed
        story_ids = [
            story_id for story_id in story_ids if story_id not in self.processed_ids
        ]

        # Limit to 40 posts per time
        story_ids = story_ids[:40]

        # Create tasks to fetch details for each story concurrently
        tasks = [self.process_post(str(story_id)) for story_id in story_ids]

        # Run the tasks concurrently
        await asyncio.gather(*tasks)
        print("finished")


if __name__ == "__main__":
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    username = "PostCreator"
    password = "passwordJohnCena55524"
    login(username, password)

    hacker_news_post_creator = HackerNewsPostCreator(output_dir, 5)
    # arxiv_post_creator = ArxivPostCreator(output_dir)

    # fetch and post every 4 hours
    while True:
        asyncio.run(hacker_news_post_creator.run())
        time.sleep(3600 * 4)
