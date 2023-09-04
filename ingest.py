import json
import os
import re
import shutil
from typing import List

import requests
from bs4 import BeautifulSoup

rubric = None
message = None
rubric_file = 'docs/rubric_data.json'
discussion_entries_file = 'docs/discussion_entries.json'

class DiscussionEntry:
    def __init__(self, id: int, parent_id: int, name: str, message: str, replies: List):
        self.id = id
        self.parent_id = parent_id
        self.name = name
        self.message = message
        self.replies = replies

    def to_json(self):
        return {
            'id': self.id,
            'parent_id': self.parent_id,
            'name': self.name,
            'message': self.message,
            'replies': [reply.to_json() for reply in self.replies]
        }

    def dump_json(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_json(), f)

def extract_entries(entries, participants):
    result = []
    counter = 0
    for entry in entries:
        if 'message' in entry and 'deleted' not in entry:
            id = entry['id']
            parent_id = entry['parent_id']
            user_id = entry['user_id']
            name = next((f"Student {counter}" for p in participants if p['id'] == user_id), None)
            message = entry['message']
            replies = []
            if 'replies' in entry:
                replies = extract_entries(entry['replies'], participants)
            result.append(DiscussionEntry(id, parent_id, name, message, replies))
            counter += 1
    return result

def save_messages(entries, group_id=None):

    for entry in entries:
        filename = f'docs/{entry.name}.html'
        if group_id is not None:
            filename = f'docs/group_{group_id}_{entry.name}.html'

        with open(filename, 'a+') as f:
            if  entry.parent_id == None:
                f.write(f'<h1><b>Student Post: {entry.name}</b></h1>')
                f.write(entry.message)
                f.write('<hr>')
            else:
                f.write(f'<h2><b>Reply to: {entry.parent_id}</b></h2>')
                f.write(entry.message)
                f.write('<hr>')

        save_messages(entry.replies, group_id)

def extract_group_discussions(group_topic_children, headers):
    group_entries = []
    for group_topic in group_topic_children:
        group_id = group_topic['group_id']
        topic_id = group_topic['id']
        group_discussion_url = f'{base_url}/api/v1/groups/{group_id}/discussion_topics/{topic_id}/view'
        group_discussion_response = requests.get(group_discussion_url, headers=headers)
        if group_discussion_response.ok:
            group_discussion_data = group_discussion_response.json()
            entries = extract_entries(group_discussion_data['view'], group_discussion_data['participants'])
            # Dump JSON data for group-based discussion
            with open(discussion_entries_file, 'w') as f:
                json.dump([entry.to_json() for entry in entries], f)
            group_entries.append({
                'group_id': group_id,
                'entries': entries
            })
    return group_entries

def extract_individual_discussion(discussion_url, headers):
    individual_entries = []
    discussion_response = requests.get(discussion_url, headers=headers)
    if discussion_response.ok:
        discussion_data = discussion_response.json()
        entries = extract_entries(discussion_data['view'], discussion_data['participants'])
        # Dump JSON data for individual discussion
        with open(discussion_entries_file, 'w') as f:
            json.dump([entry.to_json() for entry in entries], f)
        individual_entries.extend(entries)
    return individual_entries


def ingest_canvas_discussions(input_url, access_token):
    global base_url, rubric, message
    match = re.match(r'https://canvas.illinois.edu/courses/(\d+)/discussion_topics/(\d+)', input_url)
    if match:
        course_id, discussion_topic_id = match.groups()
    else:
        raise ValueError("Invalid URL")
    base_url = 'https://canvas.illinois.edu'
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    discussion_url = f'{base_url}/api/v1/courses/{course_id}/discussion_topics/{discussion_topic_id}/view'
    instruction_url = f'{base_url}/api/v1/courses/{course_id}/discussion_topics/{discussion_topic_id}'
    instruction_response = requests.get(instruction_url, headers=headers)
    if instruction_response.ok:
        instruction_data = instruction_response.json()
        print(instruction_data)
        rubric = []

        # Extract title if it exists
        if 'title' in instruction_data:
            title = instruction_data['title']
            rubric = [{'title': title}]

        if 'description' in instruction_data['assignment']:
            message_html = instruction_data['assignment']['description']
            soup = BeautifulSoup(message_html, 'html.parser')
            message = soup.get_text()
            rubric.append({'instruction': message})

        if 'rubric' in instruction_data['assignment'] and 'description' in instruction_data['assignment']:
            rubric.extend(instruction_data['assignment']['rubric'])

            if 'points_possible' in instruction_data['assignment']:
                points_possible = instruction_data['assignment']['points_possible']
                rubric.append({'points_possible': points_possible})

            # Check if the docs folder exists
            if os.path.exists('docs'):
                #delete the folder
                shutil.rmtree('docs')

            # Create the docs folder
            os.makedirs('docs')
            with open(rubric_file, 'w') as f:
                json.dump(rubric, f)

            print("Extracted instructions and rubric")
        else:
            print(f'Error: {instruction_response.text}')

        # Check if the discussion is an individual discussion with associated group-based discussions
        if 'group_topic_children' in instruction_data and len(instruction_data['group_topic_children']) > 0:
            # Extract and save group-based discussion posts
            group_entries = extract_group_discussions(instruction_data['group_topic_children'], headers)
            os.makedirs('docs', exist_ok=True)
            print("Extracted group discussion entries: {}" + str(len(group_entries)))
            for group_entry in group_entries:
                save_messages(group_entry['entries'], group_entry['group_id'])
        else:
            # Extract and save standalone individual discussion posts
            individual_entries = extract_individual_discussion(discussion_url, headers)
            print("Extracted individual discussion entries")
            os.makedirs('docs', exist_ok=True)
            save_messages(individual_entries)

    else:
        print(f'Error: {instruction_response.text}')


def create_vector_store():

    return None