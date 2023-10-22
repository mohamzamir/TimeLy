import openai
from transformers import pipeline
import pandas as pd
from transformers import DistilBertTokenizer
import os
import spacy
from flask import Flask, request, jsonify
import ast
from flask_cors import CORS

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Check if processed data exists
CACHE_FILE = "processed_data.parquet"

if os.path.exists(CACHE_FILE):
    # Load from cache
    grouped = pd.read_parquet(CACHE_FILE)
else:
    # Read data from the Excel file
    df = pd.read_excel("output_file.xlsx")


    def get_average_score(row):
        total_students = row['A'] + row['A-'] + row['B+'] + row['B'] + row['B-'] + row['C+'] + row['C'] + row['C-'] + \
                         row[
                             'D+'] + row['D'] + row['F']

        if total_students == 0:
            return 0  # or any other value you deem appropriate for this case

        average_score = (row['A'] * 4 + row['A-'] * 3.7 + row['B+'] * 3.3 + row['B'] * 3 + row['B-'] * 2.7 + row[
            'C+'] * 2.3 + row['C'] * 2 + row['C-'] * 1.7 + row['D+'] * 1.3 + row['D'] * 1) / total_students
        return average_score


    # Group by course name and instructor, then aggregate the data
    grouped = df.groupby(['Course', 'Instructor', 'Department']).agg({
        'A': 'sum', 'A-': 'sum', 'B+': 'sum', 'B': 'sum', 'B-': 'sum',
        'C+': 'sum', 'C': 'sum', 'C-': 'sum', 'D+': 'sum', 'D': 'sum', 'F': 'sum',
        'Good Comments': lambda x: ' '.join(map(str, x)),
        'Bad Comments': lambda x: ' '.join(map(str, x)),
        'Section': lambda x: list(map(str, x.unique()))
    }).reset_index()

    grouped['Average Score'] = grouped.apply(get_average_score, axis=1)

    # Create a sentiment analysis pipeline
    nlp = pipeline("sentiment-analysis")

    MAX_TOKENS = 500


    def truncate_text(text):
        tokens = tokenizer.tokenize(text)
        if len(tokens) > MAX_TOKENS:
            tokens = tokens[:MAX_TOKENS]
        return tokenizer.convert_tokens_to_string(tokens)


    def get_sentiment_score(text):
        # Truncate or split the text if it's too long
        text = truncate_text(text)
        print(f"Token count: {len(tokenizer.tokenize(text))}")
        result = nlp(text)
        sentiment = result[0]['label']
        confidence = result[0]['score']

        if sentiment == "POSITIVE":
            return confidence * 10  # Scaled to match -10 to 10
        else:
            return -confidence * 10  # Scaled to match -10 to 10


    grouped['Sentiment Score'] = grouped['Good Comments'].apply(get_sentiment_score) - grouped['Bad Comments'].apply(
        get_sentiment_score)
    # After all processing, save to cache
    grouped.to_parquet(CACHE_FILE)

# Normalize Average Score to [0, 1]
max_avg_score = grouped['Average Score'].max()
min_avg_score = grouped['Average Score'].min()
grouped['Normalized Avg Score'] = (grouped['Average Score'] - min_avg_score) / (max_avg_score - min_avg_score)

# Normalize Sentiment Score to [0, 1]
max_sentiment_score = grouped['Sentiment Score'].max()
min_sentiment_score = grouped['Sentiment Score'].min()
grouped['Normalized Sentiment Score'] = (grouped['Sentiment Score'] - min_sentiment_score) / (
        max_sentiment_score - min_sentiment_score)

# Define weights (adjust as needed)
weight_avg_score = 0.5
weight_sentiment_score = 0.5

# Calculate combined metric
grouped['Combined Metric'] = (weight_avg_score * grouped['Normalized Avg Score']) + (
        weight_sentiment_score * grouped['Normalized Sentiment Score'])

# Categorize based on the combined metric
lower_percentile = grouped['Combined Metric'].quantile(0.33)
upper_percentile = grouped['Combined Metric'].quantile(0.66)


def course_difficulty(row):
    if row['Combined Metric'] <= lower_percentile:
        return "Hard"
    elif lower_percentile < row['Combined Metric'] <= upper_percentile:
        return "Medium Hard"
    else:
        return "Easy"


grouped['Difficulty'] = grouped.apply(course_difficulty, axis=1)


# Use the helper function to determine sentiment description
def get_sentiment_description(score):
    if score > 7:
        return "This course has received overwhelmingly positive feedback."
    elif 2 < score <= 7:
        return "This course has received mostly positive feedback."
    elif -2 < score <= 2:
        return "The feedback for this course has been mixed."
    else:
        return "This course has received mostly negative feedback."


def get_courses_based_on_difficulty(difficulty, department=None, num_courses=1):
    if department:
        courses = grouped[
            (grouped['Difficulty'] == difficulty) & (grouped['Department'].str.contains(department, case=False))]
    else:
        courses = grouped[grouped['Difficulty'] == difficulty]

    if len(courses) < num_courses:  # Handle the case of not enough courses
        num_courses = len(courses)
    recommended_courses = courses.sample(n=num_courses) if num_courses > 0 else courses

    return [(row['Course'], row['Instructor'], row['Section'], row['Course Overview']) for _, row in
            recommended_courses.iterrows()]


def generate_course_overview(row):
    sentiment_description = get_sentiment_description(row['Sentiment Score'])

    good_comments = str(row['Good Comments']) if pd.notna(row['Good Comments']) else "No positive feedback provided."
    bad_comments = str(row['Bad Comments']) if pd.notna(row['Bad Comments']) else "No negative feedback mentioned."

    good_comments_summary = f"Positive Feedback: {good_comments}"
    bad_comments_summary = f"Negative Feedback: {bad_comments}"

    return f"{sentiment_description}"


grouped['Course Overview'] = grouped.apply(generate_course_overview, axis=1)

nlp = spacy.load("en_core_web_sm")


def parse_user_input_spacy(text, departments):
    doc = nlp(text)
    results = []

    # Variables to store extracted information
    current_num = 1
    current_difficulty = None
    department_name = None

    # Iterate over tokens in the document
    for token in doc:
        if token.pos_ == "NUM":
            if current_num and current_difficulty and department_name:
                results.append((current_num, current_difficulty, department_name))
                current_num = current_difficulty = department_name = None
            current_num = int(token.text)

        elif token.text.lower() in ["hard", "easy", "medium", "medium hard"]:
            current_difficulty = token.text.lower()

    # Capture department name based on noun chunks and match with available departments
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        for department in departments:
            if department.lower() in chunk_text:
                department_name = department
                break

    if current_num and current_difficulty and department_name:
        results.append((current_num, current_difficulty, department_name))
        current_num = current_difficulty = department_name = None

    if "gpa booster" in text.lower():
        results.append(("GPA Booster", None, None))

    return results


def map_to_defined_difficulties(difficulty):
    """Maps user input difficulty to one of the predefined difficulties."""
    difficulty_mapping = {
        "hard": "Hard",
        "easy": "Easy",
        "medium": "Medium Hard",
        "medium hard": "Medium Hard"
    }
    return difficulty_mapping.get(difficulty.lower(), "Medium Hard")  # Default to "Medium Hard" if not found


openai.api_key = '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'


# def assist(user_input, conversation_history=[]):
#     # cached_data = pd.read_parquet(CACHE_FILE)
#     # matched_data = cached_data[cached_data['Course'].str.contains(user_input, case=False, na=False)]
#     messages = conversation_history + [{"role": "user", "content": user_input}]
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=messages
#     )
#     return response.choices[0].message['content']

def assist(user_input, excluded_departments=[], conversation_history=[]):
    for department in excluded_departments:
        # This will replace department names with an empty string, effectively removing them from user_input
        user_input = user_input.replace(department, "")

    messages = conversation_history + [{"role": "user", "content": user_input.strip()}]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response.choices[0].message['content']


app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "HackHarvard API"

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json['user_input'].strip().replace("/\n/g","")

    available_departments = grouped['Department'].unique().tolist()
    parsed_data = parse_user_input_spacy(user_input, available_departments)

    department_queries = []
    other_queries = user_input  # Start with the full query

    queried_departments = []  # List to hold departments that were queried

    for num, difficulty, department in parsed_data:
        if department:
            department_queries.append((num, difficulty, department))
            dep_query = f"{num} {difficulty if difficulty else ''} {department} courses".strip()
            other_queries = other_queries.replace(dep_query, "").strip()
            queried_departments.append(department)  # Add the department to the queried_departments list

    department_responses = []
    other_responses = []
    for num, difficulty, department in department_queries:
        recommended_courses = get_courses_based_on_difficulty(map_to_defined_difficulties(difficulty), department, num)
        for course, instructor, sections, overview in recommended_courses:
            sections_list = ast.literal_eval(sections.decode("utf-8")) if isinstance(sections, bytes) else []
            course_code = "Unknown Code"
            if sections_list and isinstance(sections_list, list) and isinstance(sections_list[0], str):
                course_code = sections_list[0].split('-')[0]
            department_responses.append(
                f"Course: {course_code} - {course}, Instructor: {instructor}\nOverview: {overview}\n")

    if department_responses or other_queries:
        # Directly instruct the model to avoid certain departments if they are present in queried_departments
        exclusion_instruction = " ".join([f"Do not suggest any {dep} courses." for dep in queried_departments])
        modified_query = f"{other_queries} {exclusion_instruction}"
        response = assist(modified_query.strip())
        other_responses.append(response)

    # if not department_responses:
    #     response = assist(other_queries.strip())
    #     other_responses.append(response.strip())

    # Prepare the final response to be sent as JSON
    response_data = {
        "department_courses_recommendations": department_responses if department_responses else [
            "No specific department courses found based on the query."],
        "other_queries_responses": other_responses if other_responses else [
            "No other specific queries found based on the query."]
    }

    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)