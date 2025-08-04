

class Question:
    def __init__(self, question_text, corrct_answer):
        self.question_text = question_text
        self.corrct_answer = corrct_answer

question_list=  [
    Question("What is the full form of CPU?", "Central Processing Unit"),
    Question("What is the full form of RAM?", "Random Access Memory"),
    Question("What is the full form of GPU?", "Graphics Processing Unit"),
    Question("What is the full form of SSD?", "Solid State Drive"),
    Question("What is the full form of USB?", "Universal Serial Bus"),
    Question("What is the full form of LAN?", "Local Area Network"),
    Question("What is the full form of ROM?", "Read-Only Memory"),
    Question("What is the full form of BIOS?", "Basic Input/Output System"),
    Question("What is the full form of HTTP?", "HyperText Transfer Protocol"),
    Question("What is the full form of IP?", "Internet Protocol")
]

def ask_question(question):
    user_response = input(question.question_text).lower()
    if user_response == question.corrct_answer.lower():
        print("Correct answer ")
        return 1
    else:
        print("Incorrect answer ")
        return 0
def ask_user_mode():
    answer = input("Would you like to play game ? (y/n) ")
    if answer == "y":
        print("Lets start the game")
        return 1
    else:
        print("Game is terminated on user request ")
        return 0

if __name__ == '__main__':
    print("Welcome to the Quiz Game developed by Nirnajn lamichhane !")
    usermode = ask_user_mode()
    if usermode == 0 :
        quit(839443)
    while True:
        score = 0
        for questionDetail in question_list:
            score = score+ ask_question(questionDetail)
        print("Your score is: ", score)
        user_preference =input("Would you like to play game again  ? (y/n) ")
        if user_preference != "y":
            break
