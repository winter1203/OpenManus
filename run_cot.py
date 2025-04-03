#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   run_cot.py
@Time    :   2025/04/01
@Author  :   Winter.Yu
@Version :   1.0
@Contact :   winter741258@126.com
@Desc    :   None
'''

# here put the import lib
import os
from llm_api.api_base import create_api
from dataclasses import dataclass
from loguru import logger
from typing import List, Optional
import textwrap
import re
import webbrowser
from pathlib import Path
import toml


@dataclass
class CoTStep:
    """Data class representing a single CoT step"""
    number: int
    content: str

@dataclass
class CoTResponse:
    """Data class representing a complete CoT response"""
    question: str
    steps: List[CoTStep]
    answer: Optional[str] = None


@dataclass
class VisualizationConfig:
    """Configuration for CoT visualization"""
    max_chars_per_line: int = 40
    max_lines: int = 4
    truncation_suffix: str = "..."


def wrap_text(text: str, config: VisualizationConfig) -> str:
    """Wrap text to fit within box constraints"""
    text = text.replace('\n', ' ').replace('"', "'")
    wrapped_lines = textwrap.wrap(text, width=config.max_chars_per_line)

    if len(wrapped_lines) > config.max_lines:
        wrapped_lines = wrapped_lines[:config.max_lines]
        wrapped_lines[-1] = wrapped_lines[-1][:config.max_chars_per_line-3] + "..."

    return "<br>".join(wrapped_lines)


def run_cot(question, api_key, model, base_url):
    try:
        api = create_api("deepseek", api_key, model, base_url)
    except Exception as e:
        logger.error(f"Error creating API: {e}")
    logger.info(f"Generating response for question using {model}")
    try:
        raw_response = api.generate_response(
            question,
            max_tokens=2048
            )

        return raw_response
    except Exception as e:
        logger.error(f"Error generating response: {e}")



def parse_cot_response(response_text: str, question: str) -> CoTResponse:
    """
    Parse CoT response text to extract steps and final answer.

    Args:
        response_text: The raw response from the API
        question: The original question

    Returns:
        CoTResponse object containing question, steps, and answer
    """
    # Extract all steps
    step_pattern = r'<step number="(\d+)">\s*(.*?)\s*</step>'
    steps = []
    for match in re.finditer(step_pattern, response_text, re.DOTALL):
        number = int(match.group(1))
        content = match.group(2).strip()
        steps.append(CoTStep(number=number, content=content))

    # Extract answer
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    answer_match = re.search(answer_pattern, response_text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None

    # Sort steps by number
    steps.sort(key=lambda x: x.number)

    return CoTResponse(question=question, steps=steps, answer=answer)


def create_mermaid_diagram(cot_response: CoTResponse, config: VisualizationConfig) -> str:
    """
    Convert CoT steps to Mermaid diagram with improved text wrapping.

    Args:
        cot_response: CoTResponse object containing the reasoning steps
        config: VisualizationConfig for text formatting

    Returns:
        Mermaid diagram markup as a string
    """
    diagram = ['<div class="mermaid">', 'graph TD']

    # Add question node
    question_content = wrap_text(cot_response.question, config)
    diagram.append(f'    Q["{question_content}"]')

    # Add steps with wrapped text and connect them
    if cot_response.steps:
        # Connect question to first step
        diagram.append(f'    Q --> S{cot_response.steps[0].number}')

        # Add all steps
        for i, step in enumerate(cot_response.steps):
            content = wrap_text(step.content, config)
            node_id = f'S{step.number}'
            diagram.append(f'    {node_id}["{content}"]')

            # Connect steps sequentially
            if i < len(cot_response.steps) - 1:
                next_id = f'S{cot_response.steps[i + 1].number}'
                diagram.append(f'    {node_id} --> {next_id}')

    # Add final answer node
    if cot_response.answer:
        answer = wrap_text(cot_response.answer, config)
        diagram.append(f'    A["{answer}"]')
        if cot_response.steps:
            diagram.append(f'    S{cot_response.steps[-1].number} --> A')
        else:
            diagram.append('    Q --> A')

    # Add styles for better visualization
    diagram.extend([
        '    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;',
        '    classDef question fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;',
        '    classDef answer fill:#d4edda,stroke:#28a745,stroke-width:2px;',
        '    class Q question;',
        '    class A answer;',
        '    linkStyle default stroke:#666,stroke-width:2px;'
    ])

    diagram.append('</div>')
    return '\n'.join(diagram)


def main():
    file_path = "./config/config.toml"
    config = toml.load(file_path)
    api_key = config['llm']['api_key']
    model = config['llm']['model']
    base_url = config['llm']['base_url']
    html_path = "./llm_api/templates/index.html"
    if Path('./workspace/index.html').exists():
        os.remove('./workspace/index.html')
    while True:
        question = input("Enter your question, or type 'exit'/'quit'/'q' to quit: ")
        if question.strip().lower() in ["exit", "quit", "q"]:
            logger.info("Exiting...See you next time!")
            break
        raw_response = run_cot(question.strip(), api_key, model, base_url)
        result = parse_cot_response(raw_response, question)
        viz_config = VisualizationConfig(
                max_chars_per_line=40,
                max_lines=8
            )
        visualization = create_mermaid_diagram(result, viz_config)
        with open(html_path, "r") as f:
            html = f.read()
        html = html.replace("{{raw_output}}", raw_response).replace("{{mermaid_code}}", visualization)
        with open('./workspace/index.html', "w") as f:
            f.write(html)
        webbrowser.open(f'{os.getcwd()}/workspace/index.html')



if __name__ == "__main__":
    main()

