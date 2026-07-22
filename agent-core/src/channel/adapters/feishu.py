"""
channel/adapters/feishu.py — Feishu (Lark) adapter using WebSocket long connection.

Uses lark-oapi SDK's built-in WebSocket mode — outbound connection to Feishu servers,
no public IP or webhook needed. Same pattern as Telegram long-polling and Slack Socket Mode.

Requires: pip install lark-oapi
Config: {app_id, app_secret}

Required Feishu permissions:
- im:message — receive messages
- im:message:send_as_bot — send messages as bot
- im:chat:readonly — list chats (optional)

Event subscription:
- Event: im.message.receive_v1
- Mode: 长连接 (WebSocket long connection)
"""

import asyncio
import json

from channel.adapter import ChannelAdapter, InboundMessage, OutboundMessage, OnMessageCallback

# Common Feishu error codes and actionable messages
_FEISHU_ERROR_HINTS = {
    10003: 'Invalid app_id. Check your Feishu app credentials in Channel settings.',
    10014: 'Invalid app_secret. Check your Feishu app credentials in Channel settings.',
    99991663: 'Tenant token invalid. Try restarting the channel adapter.',
    99991668: 'Tenant token expired. The adapter will auto-refresh, try again.',
    99991672: 'Permission denied. Grant the required permission in Feishu Developer Console: https://open.feishu.cn/app/{app_id}/auth',
    230001: 'Bot not in this chat. Add the bot to the chat first, or the user needs to message the bot directly.',
    230002: 'Bot has been removed from chat. Re-add the bot.',
    230006: 'Message send failed: bot not activated. Publish your app version in Feishu Developer Console.',
    230014: 'Message too long. Maximum 4096 characters.',
}


class FeishuAdapter(ChannelAdapter):
    """Feishu/Lark adapter using SDK WebSocket long connection."""

    def __init__(self, channel_id: str, platform: str, config: dict,
                 on_message: OnMessageCallback):
        super().__init__(channel_id, platform, config, on_message)
        self._client = None
        self._task: asyncio.Task | None = None
        self._api_client = None

    async def start(self) -> None:
        app_id = self.config.get('app_id', '')
        app_secret = self.config.get('app_secret', '')
        if not app_id or not app_secret:
            raise ValueError(
                'Feishu app_id and app_secret are required. '
                'Configure them in Settings → Channels.'
            )

        import lark_oapi as lark

        # Store the main event loop for cross-thread scheduling
        self._loop = asyncio.get_event_loop()

        # Create API client for sending messages
        self._api_client = lark.Client.builder() \
            .app_id(app_id) \
            .app_secret(app_secret) \
            .build()

        # Create event handler
        event_handler = lark.EventDispatcherHandler.builder("", "") \
            .register_p2_im_message_receive_v1(self._handle_message_event) \
            .build()

        # Create WebSocket long connection client
        self._client = lark.ws.Client(
            app_id=app_id,
            app_secret=app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.INFO,
        )

        # Start in background thread (SDK blocks)
        self._task = asyncio.create_task(self._run())
        self._running = True
        print(f'[feishu] adapter started (WebSocket mode): {self.channel_id}')

    async def _run(self):
        """Run the lark WebSocket client in a dedicated thread with its own event loop."""
        import threading

        def _thread_target():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                import lark_oapi.ws.client as ws_mod
                ws_mod.loop = new_loop
                self._client.start()
            except Exception as e:
                err_msg = str(e)
                if 'invalid' in err_msg.lower() and ('app_id' in err_msg.lower() or 'secret' in err_msg.lower()):
                    print(f'[feishu] Connection failed: invalid app credentials. '
                          f'Check app_id and app_secret in Channel settings.')
                else:
                    print(f'[feishu] WebSocket connection error: {e}')
                self._running = False
            finally:
                new_loop.close()

        self._thread = threading.Thread(target=_thread_target, daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._client = None
        print(f'[feishu] adapter stopped: {self.channel_id}')

    async def send_message(self, msg: OutboundMessage) -> None:
        """Send message via Feishu Open API."""
        if not self._api_client:
            raise RuntimeError(
                '[feishu] Cannot send: adapter not initialized. '
                'Check app_id/app_secret in Channel settings.'
            )

        import lark_oapi as lark
        from lark_oapi.api.im.v1 import CreateMessageRequest, CreateMessageRequestBody

        body = CreateMessageRequestBody.builder() \
            .receive_id(msg.chat_id) \
            .msg_type('text') \
            .content(json.dumps({'text': msg.text})) \
            .build()

        request = CreateMessageRequest.builder() \
            .receive_id_type('chat_id') \
            .request_body(body) \
            .build()

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None, self._api_client.im.v1.message.create, request
            )
            if not response.success():
                hint = _FEISHU_ERROR_HINTS.get(response.code, '')
                app_id = self.config.get('app_id', '')
                if hint:
                    hint = hint.format(app_id=app_id)
                error_detail = f'[feishu] send_message failed (code={response.code}): {response.msg}'
                if hint:
                    error_detail += f'\n  → {hint}'
                print(error_detail)
                raise RuntimeError(error_detail)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f'[feishu] send_message exception: {e}')

    def _handle_message_event(self, data):
        """Handle im.message.receive_v1 event from SDK callback (runs in thread)."""
        try:
            event = data.event
            message = event.message
            sender = event.sender

            # Skip bot messages
            if sender.sender_type == 'app':
                return

            # Extract text
            msg_type = message.message_type
            text = ''
            if msg_type == 'text':
                content = json.loads(message.content)
                text = content.get('text', '')
            else:
                text = f'[{msg_type}]'

            if not text:
                return

            sender_id = sender.sender_id.open_id or sender.sender_id.user_id or ''
            chat_id = message.chat_id

            msg = InboundMessage(
                platform='feishu',
                channel_id=self.channel_id,
                user_id=sender_id,
                chat_id=chat_id,
                display_name=sender_id,
                text=text,
            )

            # Schedule coroutine from SDK thread to main event loop
            asyncio.run_coroutine_threadsafe(self._on_message(msg), self._loop)

        except Exception as e:
            print(f'[feishu] handle message error: {e}')
            print(f'  If messages are not being received, ensure:')
            print(f'  1. Event "im.message.receive_v1" is subscribed in Feishu Developer Console')
            print(f'  2. Subscription mode is set to "长连接" (WebSocket long connection)')
            print(f'  3. App has "im:message" permission and is published')
