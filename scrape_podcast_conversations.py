"""
Data Scraping and Cleaning Script
================================

Description:
    This script scrapes all podcast transcripts from the Analytics Power Hour podcast website
    by automating a web browser session, parsing the page content, and storing the
    extracted information in a structured format. 
    
    This script extracts information (title, episode number, publication date, summary, participants, and transcript)
    from a preset list of podcast episode URLs. It uses Selenium to automate browser interactions,
    BeautifulSoup for parsing, and regular expressions for cleaning. Finally, it saves
    the resulting data in a pandas DataFrame and writes it to a pickle file.

QC Steps (Data Cleaning and Transformation):
    • Remove redundant episode intro and exit prerecorded content
    • Remove blank lines and announcer lines
    • Remove ad content
    • Replace short forms of speaker names with full names
    • Extract a list of unique participants
"""

import time
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import ElementClickInterceptedException, NoSuchElementException
from bs4 import BeautifulSoup

# ------------------------------------------------------------------------------
# Configuration: Update file paths and constants here
# ------------------------------------------------------------------------------
url_csv_path = "/Users/brianmiller/Desktop/analytics_power_hour_chatbot/data_episode_transcripts/podcast_urls.csv"
# ------------------------------------------------------------------------------

def read_urls_into_list(csv_path, url_column="url"):
    """
    Reads the CSV file at csv_path and returns a list of URLs from url_column.
    """
    df = pd.read_csv(csv_path)
    return df[url_column].tolist()

# Load in the URLs
url_list = read_urls_into_list(url_csv_path)

print(f"Extracting conversations from {len(url_list)} podcast episodes")

def extract_info_from_url(url):
    """
    Extracts podcast information (title, publication date, summary, transcript)
    from a single URL using Selenium and BeautifulSoup.
    """
    # Set up the Selenium WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)

    # Wait for the page to load completely
    time.sleep(2)

    # Attempt to close any pop-ups or overlays
    try:
        close_button = driver.find_element(By.XPATH, 'XPATH_OF_CLOSE_BUTTON')
        close_button.click()
        time.sleep(2)
    except NoSuchElementException:
        pass

    # Find and click the transcript button (or any relevant button) using XPath
    try:
        button = driver.find_element(By.XPATH, '/html/body/div[2]/div/div[1]/div/div/div[6]/div[1]/i')
        button.click()
    except ElementClickInterceptedException:
        # If the click is intercepted, scroll to the element and try again
        driver.execute_script("arguments[0].scrollIntoView(true);", button)
        time.sleep(1)
        button.click()

    # Wait for the content to load
    time.sleep(2)

    # Parse the page source
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'html.parser')

    # Extract specific fields
    try:
        title = driver.find_element(By.XPATH, '/html/body/div[2]/div/div[1]/div/div/div[1]/div/div/div/div/div[1]/h1').text
    except NoSuchElementException:
        title = None

    try:
        date_published = driver.find_element(By.XPATH, '/html/body/div[2]/div/div[1]/div/div/div[1]/div/div/div/div/div[4]/p').text
    except NoSuchElementException:
        date_published = None

    try:
        summary = driver.find_element(By.XPATH, '/html/body/div[2]/div/div[1]/div/div/div[4]/div/p').text
    except NoSuchElementException:
        summary = None

    try:
        transcript = driver.find_element(By.XPATH, '/html/body/div[2]/div/div[1]/div/div/div[6]/div[2]').text
    except NoSuchElementException:
        transcript = None

    # Close the WebDriver
    driver.quit()

    return {
        'Title': title,
        'URL': url,
        'Date Published': date_published,
        'Summary': summary,
        'Transcript': transcript
    }

# ------------------------------------------------------------------------------
# Main Data Extraction Loop
# ------------------------------------------------------------------------------
data = []
for url in url_list:
    print(f"Extracting data from: {url}")
    info = extract_info_from_url(url)
    data.append(info)

df = pd.DataFrame(data)

# ------------------------------------------------------------------------------
# Data Cleaning and Transformation
# ------------------------------------------------------------------------------
# Remove the "Published: " prefix from the "Date Published" column
df['Date Published'] = df['Date Published'].str.split(': ').str[1]

# Create a new column 'episode number' by extracting the episode number from the 'Title'
df['episode number'] = df['Title'].str.extract(r"^(?:Episode\s*)?#(\d+)")

# Remove the first 6 characters from the 'Title' column
df['Title'] = df['Title'].str.replace(r"^(?:Episode\s*)?#\d+\s*:\s*", "", regex=True)

def remove_brackets(text):
    """
    Remove all content within square brackets from the given text.
    """
    return re.sub(r'\[.*?\]', '', text)

df['Transcript'] = df['Transcript'].apply(remove_brackets)

def remove_blank_lines(text):
    """
    Remove all blank lines from the given text.
    """
    return "\n".join([line for line in text.split("\n") if line.strip()])

df['Transcript'] = df['Transcript'].apply(remove_blank_lines)

def remove_announcer_lines(text):
    """
    Remove all lines containing Announcer references from the given text.
    """
    lines = text.split('\n')
    filtered_lines = [line for line in lines if "(Announcer):" not in line and "Announcer:" not in line]
    return '\n'.join(filtered_lines)

df['Transcript'] = df['Transcript'].apply(remove_announcer_lines)

def remove_last_five_lines(text):
    """
    Remove the last five lines from the given text.
    """
    lines = text.split("\n")
    return "\n".join(lines[:-6])

df['Transcript'] = df['Transcript'].apply(remove_last_five_lines)

def replace_names(transcript):
    """
    Clean participant names in the transcript.
    """
    transcript = transcript.replace('Michael:', 'Michael Helbling:')
    transcript = transcript.replace('MH:', 'Michael Helbling:')
    transcript = transcript.replace('Moe:', 'Moe Kiss:')
    transcript = transcript.replace('MK:', 'Moe Kiss:')
    transcript = transcript.replace('Josh:', 'Josh Crowhurst:')
    transcript = transcript.replace('JC:', 'Josh Crowhurst:')
    transcript = transcript.replace('Tim:', 'Tim Wilson:')
    transcript = transcript.replace('TW:', 'Tim Wilson:')
    transcript = transcript.replace('JH:', 'Julie Hoyer:')
    transcript = transcript.replace('Julie:', 'Julie Hoyer:')
    transcript = transcript.replace('S1:', 'Unknown:')
    transcript = transcript.replace('Speaker 1:', 'Unknown:')
    transcript = transcript.replace('S2:', 'Unknown:')
    transcript = transcript.replace('Speaker 2:', 'Unknown:')
    transcript = transcript.replace('S3:', 'Unknown:')
    transcript = transcript.replace('Speaker 3:', 'Unknown:')
    transcript = transcript.replace('S4:', 'Unknown:')
    transcript = transcript.replace('Speaker 4:', 'Unknown:')
    transcript = transcript.replace('S5:', 'Unknown:')
    transcript = transcript.replace('Speaker 5:', 'Unknown:')
    transcript = transcript.replace('S6:', 'Unknown:')
    transcript = transcript.replace('Speaker 6:', 'Unknown:')
    transcript = transcript.replace('S7:', 'Unknown:')
    transcript = transcript.replace('Speaker 7:', 'Unknown:')
    transcript = transcript.replace('S8:', 'Unknown:')
    transcript = transcript.replace('Speaker 8:', 'Unknown:')
    transcript = transcript.replace('S9:', 'Unknown:')
    transcript = transcript.replace('Speaker 9:', 'Unknown:')
    transcript = transcript.replace('S10:', 'Unknown:')
    transcript = transcript.replace('Speaker 10:', 'Unknown:')
    transcript = transcript.replace('S?:', 'Unknown:')
    return transcript

df['Transcript'] = df['Transcript'].apply(replace_names)

def extract_participants(transcript, episode_number):
    """
    Extract unique participants (speaker names) from a transcript, excluding 'Unknown' and 'Transcript'.
    If the episode number is between '001' and '050', returns 'Not in data' since no participant data is available.
    """
    if episode_number in [f"{i:03d}" for i in range(1, 51)]:
        return "Not in data"
    
    participants = set()
    for line in transcript.split('\n'):
        words = line.split(' ')
        if len(words) > 1:
            participant = ' '.join(words[1:]).split(':')[0].strip()
            if participant.lower() not in ["unknown", "transcript"]: # Exclude 'Unknown' and 'Transcript'`
                participants.add(participant)
    return list(participants)

# Apply the updated function to the DataFrame
df['Participants'] = df.apply(lambda row: extract_participants(row['Transcript'], row['episode number']), axis=1)

# Display the updated DataFrame (in an interactive environment)
print("Data cleaning and transformation steps completed.")

# ------------------------------------------------------------------------------
# Save to Pickle
# ------------------------------------------------------------------------------
df.to_pickle("podcast_data.pkl")
print("Data for {} podcasts have been successfully scraped, cleaned, and stored to podcast_data.pkl.".format(len(df)))
