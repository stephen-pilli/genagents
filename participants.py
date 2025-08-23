columns = [
    "Age",
    "Sex",
    "Ethnicity simplified",
    "Country of birth",
    "Country of residence",
    "Nationality",
    "Language",
    "Student status",
    "Employment status"
]

sqb_markers = ["National Highway Safety", "You are a serious reader", "When evaluating teaching job offers,"]

class Participant:

    transcript = ""
    transcript_with_response_times = ""
    choice_problem = ""

    def process_transcript(self, chat_transcript):
        # Transcript without response times
        transcript = ""
        for msg in chat_transcript:
            if any(marker in msg['content'] for marker in sqb_markers):
                break
            transcript += f"{msg['role']}: {msg['content']}\n"

        # Transcript with response times
        transcript_with_response_times = ""
        for i, msg in enumerate(chat_transcript):

            if any(marker in msg['content'] for marker in sqb_markers):
                break

            time_diff = None

            if i > 1:
                if msg['timestamp']['client_timestamp'] != 0 and chat_transcript[i-1]['timestamp']['client_timestamp'] != 0:
                    time_diff = msg['timestamp']['client_timestamp'] - chat_transcript[i-1]['timestamp']['client_timestamp']
                else:
                    try:
                        time_diff = msg['timestamp']['server_timestamp'] - chat_transcript[i-1]['timestamp']['server_timestamp']
                    except KeyError:
                        time_diff = -1

            # Add a blank line before each message except the first
            if i > 0:
                transcript_with_response_times += "\n"

            if msg['role'] == 'user' and i > 1:
                transcript_with_response_times += f"{msg['role']}: {msg['content']} (response time: {time_diff} ms)"
            else:
                transcript_with_response_times += f"{msg['role']}: {msg['content']}"

        # Extract the choice problem statement
        choice_problem = "\n".join(f"{msg['role']}: {msg['content']}" for msg in chat_transcript if any(marker in msg['content'] for marker in sqb_markers))
        choice_problem = choice_problem.replace("assistant: ", "")

        self.transcript = transcript
        self.transcript_with_response_times = transcript_with_response_times
        self.choice_problem = choice_problem


    def __init__(self, participant_id = None, condition = None, prolific_data = None, chat_transcript = None, change_username = None, change_assistantname = None):
        # Unique identifier for the participant
        self.Participant_ID = participant_id

        # Experiment condition
        self.condition = condition

        # Demographic info from Prolific
        self.prolific_data = prolific_data

        # Chat Data
        self.chat_transcript = chat_transcript if chat_transcript is not None else []

        if self.chat_transcript is not None:
            self.process_transcript(self.chat_transcript)

        if change_username is not None:
            self.change_user_name(change_username)

        if change_assistantname is not None:
            self.change_assistant_name(change_assistantname)

    def update_participant_id(self, participant_id):
        self.Participant_ID = participant_id

    def update_condition(self, condition):
        self.condition = condition

    def update_prolific_data(self, key, value):
        if self.prolific_data is not None:
            self.prolific_data[key] = value

    def update_chat_transcript(self, chat_transcript):
        self.chat_transcript = self.process_transcript(chat_transcript)

    def get_participant_id(self):
        return self.Participant_ID

    def get_demographics(self):
        return self.prolific_data
    
    def get_messages(self):
        transcript = []
        for msg in self.chat_transcript:
            if any(marker in msg['content'] for marker in sqb_markers):
                break
            msg_copy = dict(msg)
            if 'timestamp' in msg_copy:
                del msg_copy['timestamp']
            transcript.append(msg_copy)
        
        return transcript

    def get_transcript(self) -> str:
        return self.transcript
    
    def get_transcript_with_response_times(self) -> str:
        return self.transcript_with_response_times

    def get_choice_problem(self) -> str:
        return self.choice_problem
    
    def change_user_name(self, new_name):
        self.transcript = self.transcript.replace("user:", f"{new_name}:")
        self.transcript_with_response_times = self.transcript_with_response_times.replace("user:", f"{new_name}:")
    
    def change_assistant_name(self, new_name):
        self.transcript = self.transcript.replace("assistant:", f"{new_name}:")
        self.transcript_with_response_times = self.transcript_with_response_times.replace("assistant:", f"{new_name}:")

    def get_messages_with_switch_roles(self, chat_transcript=None, 
                    old_user = "user", 
                    new_user = "user", 
                    old_assistant = "assistant", 
                    new_assistant = "assistant"):
        if chat_transcript is None:
            chat_transcript = self.get_messages()

        for i, message in enumerate(chat_transcript):
            if message["role"] == old_user:
                message["role"] = new_user
                chat_transcript[i] = message
            elif message["role"] == old_assistant:
                message["role"] = new_assistant
                chat_transcript[i] = message

        return chat_transcript

