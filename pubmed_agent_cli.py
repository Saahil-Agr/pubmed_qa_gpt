import argparse
from chat_agent import OpenAIPubMedAgent
import openai

if __name__ == "__main__":
    """
    $ python $BASE_DIR/pubmed_qa_bot/pubmed_agent_cli.py \
      --api_key dummy_key
      --state 0 \
      --question "What is the role of chemokines and chemokine receptors by lymphatic vessels (LVs) in leukocyte traffic and cancer metastasis?"
    """

    parser = argparse.ArgumentParser(description="A command Line AI Agent to interact with thousands of medical research papers sourced from pubmed")
    parser.add_argument("--api_key")
    parser.add_argument("--question")
    parser.add_argument("--state", type=int, default=0,
                        help="0: New unrelated question and requires retrieving new relevant papers"
                             "\n1: Continue asking quesiton about the last retrieved paper"
                             "\n2: Exit Chat"
                        )
    args = parser.parse_args()
    openai.api_key = args.api_key
    agent = OpenAIPubMedAgent()
    chat_history = []
    while args.state != 2:
        if args.state == 1:
            question = input(">> ")
            response, chat_history = agent.run(question=question, chat_history=chat_history, state=args.state)
            print(response)
        else:
            if args.question:
                response, chat_history = agent.run(question=args.question, chat_history=chat_history, state=args.state)
                print(response)
                args.question = ''
            else:
                question = input(">> ")
                response, chat_history = agent.run(question=question, chat_history=chat_history, state=args.state)
                print(response)

        args.state = int(input("Enter the desired state, to exit enter 2>> "))
