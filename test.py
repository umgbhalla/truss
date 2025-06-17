import asyncio
import json
from contextlib import AsyncExitStack
from dataclasses import asdict
from typing import Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_tool_call_param import Function
from openai.types.shared_params.function_definition import FunctionDefinition
from dataclasses import dataclass, field
from typing import Literal

from openai import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionReasoningEffort

from pprint import pprint
import logging  # NEW_IMPORT_FOR_LOGGING
import argparse
import os

load_dotenv()

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")  # NEW_LOGGING_CONFIG

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""

    command: str
    args: list[str]
    env: dict[str, str] | None = None
    description: str | None = None
    enabled: bool = True


@dataclass
class MCPClientConfig:
    """Configuration for the MCP client"""

    mcpServers: dict[str, MCPServerConfig] = field(default_factory=dict)


@dataclass
class LLMClientConfig:
    """Configuration for the OpenAI (compatible) client."""

    api_key: str | None = None
    organization: str | None = None
    project: str | None = None
    base_url: str | None = None


@dataclass
class LLMRequestConfig:
    """Configuration for OpenAI client.chat.completions.create.

    Notes:
        - messages are not part of LLM request config
        - tools/tool_choice are handle directly by the MCP client
    """

    model: str
    temperature: float | NotGiven | None = NOT_GIVEN
    top_p: float | NotGiven | None = NOT_GIVEN
    max_tokens: int | NotGiven | None = NOT_GIVEN
    reasoning_effort: ChatCompletionReasoningEffort | NotGiven = NOT_GIVEN
    stream: NotGiven | Literal[False] | None = NOT_GIVEN
    stop: str | list[str] | NotGiven | None = NOT_GIVEN
    seed: int | NotGiven | None = NOT_GIVEN
    presence_penalty: float | NotGiven | None = NOT_GIVEN
    frequency_penalty: float | NotGiven | None = NOT_GIVEN
    logit_bias: dict[str, int] | NotGiven | None = NOT_GIVEN

class MCPClient:
    def __init__(
        self,
        mpc_client_config: MCPClientConfig = MCPClientConfig(),
        llm_client_config: LLMClientConfig = LLMClientConfig(),
        llm_request_config: LLMRequestConfig = LLMRequestConfig("gpt-4o"),
    ):
        self.mpc_client_config = mpc_client_config
        self.llm_client_config = llm_client_config
        self.llm_request_config = llm_request_config
        self.llm_client = AsyncOpenAI(**asdict(self.llm_client_config))
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.current_server = None

        logging.info("CLIENT CREATED")  # REPLACED_PRINT_WITH_LOG

    async def connect_to_server(self, server_name: str):
        """Connect to an MCP server using its configuration name"""

        if server_name not in self.mpc_client_config.mcpServers:
            raise ValueError(
                f"Server '{server_name}' not found in MCP client configuration"
            )

        mcp_server_config = self.mpc_client_config.mcpServers[server_name]
        if not mcp_server_config.enabled:
            raise ValueError(f"Server '{server_name}' is disabled")

        stdio_server_params = StdioServerParameters(**asdict(mcp_server_config))

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(stdio_server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()  # type: ignore
        self.current_server = server_name

        # List available tools
        response = await self.session.list_tools()  # type: ignore
        logging.info("CLIENT CONNECT to %s", server_name)
        logging.info("AVAILABLE TOOLS: %s", [tool.name for tool in response.tools])

    async def process_tool_call(self, tool_call) -> ChatCompletionToolMessageParam:
        match tool_call.type:
            case "function":
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                call_tool_result = await self.session.call_tool(tool_name, tool_args)  # type: ignore

                if call_tool_result.isError:
                    raise ValueError("An error occurred while calling the tool.")

                results = []
                for result in call_tool_result.content:
                    match result.type:
                        case "text":
                            results.append(result.text)
                        case "image":
                            raise NotImplementedError("Image content is not supported")
                        case "resource":
                            raise NotImplementedError(
                                "Embedded resource is not supported"
                            )
                        case _:
                            raise ValueError(f"Unknown content type: {result.type}")

                return ChatCompletionToolMessageParam(
                    role="tool",
                    content=json.dumps({**tool_args, tool_name: results}),
                    tool_call_id=tool_call.id,
                )

            case _:
                raise ValueError(f"Unknown tool call type: {tool_call.type}")

    async def process_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        llm_request_config: LLMRequestConfig | None = None,
    ) -> list[ChatCompletionMessageParam]:
        # Set up tools and LLM request config
        if not self.session:
            raise RuntimeError("Not connected to any server")
        tools = [
            ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description if tool.description else "",
                    parameters=tool.inputSchema,
                ),
            )
            for tool in (await self.session.list_tools()).tools
        ]
        llm_request_config = LLMRequestConfig(
            **{
                **asdict(self.llm_request_config),
                **(asdict(llm_request_config) if llm_request_config else {}),
            }
        )

        last_message_role = messages[-1]["role"]

        match last_message_role:
            case "user":
                response = await self.llm_client.chat.completions.create(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    **asdict(llm_request_config),
                )
                finish_reason = response.choices[0].finish_reason

                match finish_reason:
                    case "stop":
                        messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant",
                                content=response.choices[0].message.content,
                            )
                        )
                        return messages

                    case "tool_calls":
                        tool_calls = response.choices[0].message.tool_calls
                        assert tool_calls is not None
                        messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant",
                                tool_calls=[
                                    ChatCompletionMessageToolCallParam(
                                        id=tool_call.id,
                                        function=Function(
                                            arguments=tool_call.function.arguments,
                                            name=tool_call.function.name,
                                        ),
                                        type=tool_call.type,
                                    )
                                    for tool_call in tool_calls
                                ],
                            )
                        )
                        tasks = [
                            asyncio.create_task(self.process_tool_call(tool_call))
                            for tool_call in tool_calls
                        ]
                        messages.extend(await asyncio.gather(*tasks))
                        return await self.process_messages(messages, llm_request_config)
                    case "length":
                        raise ValueError("Length limit reached")
                    case "content_filter":
                        raise ValueError("Content filter triggered")
                    case "function_call":
                        raise NotImplementedError("Function call not implemented")
                    case _:
                        raise ValueError(f"Unknown finish reason: {finish_reason}")

            case "assistant":
                # NOTE: the only purpose of this case is to trigger other tool
                # calls based on the results of the previous tool calls
                response = await self.llm_client.chat.completions.create(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    **asdict(llm_request_config),
                )
                finish_reason = response.choices[0].finish_reason

                match finish_reason:
                    case "stop":
                        # NOTE: we do not add the last response message
                        return messages

                    case "tool_calls":
                        tool_calls = response.choices[0].message.tool_calls
                        assert tool_calls is not None
                        messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant",
                                tool_calls=[
                                    ChatCompletionMessageToolCallParam(
                                        id=tool_call.id,
                                        function=Function(
                                            arguments=tool_call.function.arguments,
                                            name=tool_call.function.name,
                                        ),
                                        type=tool_call.type,
                                    )
                                    for tool_call in tool_calls
                                ],
                            )
                        )
                        results_messages = [
                            await self.process_tool_call(tool_call)
                            for tool_call in tool_calls
                        ]
                        messages.extend(results_messages)
                        return await self.process_messages(messages, llm_request_config)
                    case "length":
                        raise ValueError("Length limit reached")
                    case "content_filter":
                        raise ValueError("Content filter triggered")
                    case "function_call":
                        raise NotImplementedError("Function call not implemented")
                    case _:
                        raise ValueError(f"Unknown finish reason: {finish_reason}")

            case "tool":
                response = await self.llm_client.chat.completions.create(
                    messages=messages,
                    **asdict(llm_request_config),
                )
                finish_reason = response.choices[0].finish_reason

                match finish_reason:
                    case "stop":
                        messages.append(
                            ChatCompletionAssistantMessageParam(
                                role="assistant",
                                content=response.choices[0].message.content,
                            )
                        )

                        return await self.process_messages(messages, llm_request_config)
                    case "tool_calls":
                        raise ValueError(
                            "The message following a tool message cannot be a tool call"
                        )
                    case "length":
                        raise ValueError("Length limit reached")
                    case "content_filter":
                        raise ValueError("Content filter triggered")
                    case "function_call":
                        raise NotImplementedError("Function call not implemented")
                    case _:
                        raise ValueError(f"Unknown finish reason: {finish_reason}")

            case "developer":
                raise NotImplementedError("Developer messaages are not supported")
            case "system":
                raise NotImplementedError("System messages are not supported")
            case "function":
                raise NotImplementedError("System messages are not supported")
            case _:
                raise ValueError(f"Invalid message role: {last_message_role}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    parser = argparse.ArgumentParser(description="Run MCP client test script")
    parser.add_argument(
        "--server",
        default="sqlite",
        help="Name of the server to connect to (default: sqlite)",
    )
    parser.add_argument(
        "--query",
        default="what are tables present in my db?",
        help="User query to send to the LLM (default question lists tables)",
    )
    args = parser.parse_args()

    # Define the MCP servers configuration (can be extended in the future)
    mcp_client_config = MCPClientConfig(
        mcpServers={
            "sqlite": MCPServerConfig(
                command="uvx",
                args=["mcp-server-sqlite", "--db-path", "truss.db"],
            )
        }
    )

    # Prefer environment variable for API key to avoid hard-coding secrets
    openrouter_api_key = os.getenv(
        "OPENROUTER_API_KEY",
        "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",
    )

    # LLM configuration targeting OpenRouter (user can set OPENROUTER_API_KEY)
    llm_client_config = LLMClientConfig(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Base request configuration for the LLM
    llm_request_config = LLMRequestConfig(model="openai/gpt-4.1-mini")

    client = MCPClient(
        mcp_client_config,
        llm_client_config,
        llm_request_config,
    )

    # Connect to the requested server
    await client.connect_to_server(args.server)

    # Build the initial user message
    messages_in = [{"role": "user", "content": args.query}]

    # Process messages via LLM tools pipeline
    messages_out = await client.process_messages(messages_in)
    pprint(messages_out)

    # Pretty-print only the final assistant response for quick reference
    last_msg = messages_out[-1] if messages_out else None
    if last_msg and last_msg.get("role") == "assistant":
        content = last_msg.get("content")
        if isinstance(content, str):
            print("\nAssistant response:\n" + content)
        else:
            # In case of structured tool_calls messages
            print("\nAssistant response (structured):\n" + str(content))

    # Clean up async resources
    await client.cleanup()



if __name__ == "__main__":
    import anyio

    try:
        anyio.run(main)
    except KeyboardInterrupt:
        logging.info("Execution interrupted by user. Exiting gracefully.")

