"use client";

import {
  Send,
  Sparkles
} from "lucide-react";

import {
  useEffect,
  useRef,
  useState,
  type Dispatch,
  type SetStateAction
} from "react";

import { motion } from "framer-motion";

// =====================================================
// STREAMING TEXT
// =====================================================

function StreamingText({
  text
}: {
  text: string;
}) {

  const [visible, setVisible] =
    useState(
      text.split(" ")[0] || ""
    );

  useEffect(() => {

    const words =
      text.split(" ");

    if (words.length <= 1) {
      return;
    }

    let index = 1;

    const interval =
      setInterval(() => {

        index++;

        setVisible(

          words
            .slice(0, index)
            .join(" ")
        );

        if (
          index >= words.length
        ) {

          clearInterval(interval);
        }

      }, 18);

    return () =>
      clearInterval(interval);

  }, [text]);

  return <span>{visible}</span>;
}

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type Props = {
  chatMessages: ChatMessage[];
  chatInput: string;
  setChatInput: Dispatch<
    SetStateAction<string>
  >;
  sendChat: () => void;
  chatLoading: boolean;
};

export default function AITab({
  chatMessages,
  chatInput,
  setChatInput,
  sendChat,
  chatLoading
}: Props) {

  // =====================================================
  // AUTO SCROLL
  // =====================================================

  const bottomRef =
    useRef<HTMLDivElement | null>(null);

  useEffect(() => {

    bottomRef.current?.scrollIntoView({

      behavior: "smooth"
    });

  }, [chatMessages, chatLoading]);

  // =====================================================
  // QUICK PROMPTS
  // =====================================================

  const prompts = [

    "Is this area safe at night?",

    "Explain the urban profile",

    "Does this feel commuter-heavy?",

    "What kind of activity happens here?",

    "Best visiting time?"
  ];

  // =====================================================
  // UI
  // =====================================================

  return (

    <div className="
      relative

      flex
      flex-col

      h-full
    ">

      {/* CHAT AREA */}

      <div className="
        flex-1

        overflow-y-auto
        no-scrollbar
      ">

        {/* EMPTY STATE */}

        {

          chatMessages.length === 0 && (

            <div className="
              h-full

              flex
              flex-col

              items-center
              justify-center

              text-center
            ">

              {/* AI ORB */}

              <motion.div

                animate={{
                  scale: [1, 1.04, 1]
                }}

                transition={{
                  repeat: Infinity,
                  duration: 3
                }}

                className="
                  relative

                  w-20
                  h-20

                  rounded-full

                  bg-cyan-400/10

                  border
                  border-cyan-300/20

                  flex
                  items-center
                  justify-center

                  shadow-[0_0_60px_rgba(34,211,238,0.18)]
                "
              >

                <div className="
                  absolute
                  inset-0

                  rounded-full

                  bg-cyan-400/10

                  blur-2xl
                " />

                <Sparkles
                  size={28}
                  className="
                    relative
                    text-cyan-300
                  "
                />

              </motion.div>

              {/* TITLE */}

              <h2 className="
                mt-8

                text-[26px]

                font-semibold

                tracking-tight
              ">

                Ask UrbanMind

              </h2>

              {/* DESCRIPTION */}

              <p className="
                mt-3

                max-w-[280px]

                text-[14px]

                text-white/42

                leading-[1.8]

                font-[
                  -apple-system,
                  BlinkMacSystemFont,
                  'SF Pro Display',
                  sans-serif
                ]
              ">

                Ask about safety, movement,
                street activity, commuting patterns,
                and what the area feels like.

              </p>

              {/* QUICK PROMPTS */}

              <div className="
                mt-8

                flex
                flex-wrap

                justify-center

                gap-2
              ">

                {prompts.map((prompt) => (

                  <button
                    key={prompt}

                    onClick={() =>
                      setChatInput(prompt)
                    }

                    className="
                      px-4
                      py-2.5

                      rounded-full

                      bg-white/[0.04]

                      border
                      border-white/10

                      text-[12px]

                      text-white/60

                      transition-all

                      hover:bg-white/[0.07]

                      hover:text-white
                    "
                  >

                    {prompt}

                  </button>

                ))}

              </div>

            </div>
          )
        }

        {/* CHAT MESSAGES */}

        {

          chatMessages.length > 0 && (

            <div className="
              space-y-4
              pb-4
            ">

              {chatMessages.map(
                (msg, index) => (

                  <motion.div
                    key={index}

                    initial={{
                      opacity: 0,
                      y: 10
                    }}

                    animate={{
                      opacity: 1,
                      y: 0
                    }}

                    transition={{
                      duration: 0.2
                    }}

                    className={`

                      rounded-[24px]

                      px-4
                      py-4

                      text-[14px]

                      leading-[1.9]

                      font-[
                        -apple-system,
                        BlinkMacSystemFont,
                        'SF Pro Display',
                        sans-serif
                      ]

                      ${
                        msg.role === "user"

                        ? `
                          ml-10

                          bg-cyan-400/14

                          border
                          border-cyan-400/20

                          text-white
                        `

                        : `
                          mr-10

                          bg-white/[0.035]

                          border
                          border-white/10

                          text-white/82

                          shadow-[0_0_40px_rgba(255,255,255,0.02)]
                        `
                      }
                    `}
                  >

                    {

                      msg.role === "assistant"

                      ? (
                          <StreamingText
                            key={msg.content}
                            text={msg.content}
                          />
                        )

                      : msg.content
                    }

                  </motion.div>
                )
              )}

              {/* LOADING */}

              {

                chatLoading && (

                  <div className="
                    mr-10

                    rounded-[24px]

                    border
                    border-white/10

                    bg-white/[0.035]

                    px-4
                    py-4
                  ">

                    <div className="
                      flex
                      items-center
                      gap-2
                    ">

                      <div className="
                        w-2
                        h-2

                        rounded-full

                        bg-cyan-300

                        animate-bounce
                      " />

                      <div className="
                        w-2
                        h-2

                        rounded-full

                        bg-cyan-300

                        animate-bounce

                        [animation-delay:150ms]
                      " />

                      <div className="
                        w-2
                        h-2

                        rounded-full

                        bg-cyan-300

                        animate-bounce

                        [animation-delay:300ms]
                      " />

                    </div>

                    <p className="
                      mt-3

                      text-[12px]

                      text-white/40
                    ">

                      Generating urban intelligence...

                    </p>

                  </div>
                )
              }

              <div ref={bottomRef} />

            </div>
          )
        }

      </div>

      {/* INPUT BAR */}

      <div className="
        mt-4

        rounded-[24px]

        border
        border-white/10

        bg-white/[0.03]

        backdrop-blur-xl

        p-2

        flex
        items-center

        gap-2

        shadow-[0_0_40px_rgba(255,255,255,0.02)]
      ">

        <input

          value={chatInput}

          onChange={(e) =>
            setChatInput(
              e.target.value
            )
          }

          onKeyDown={(e) => {

            if (e.key === "Enter") {

              sendChat();
            }
          }}

          placeholder="Ask urban intelligence..."

          className="
            flex-1

            bg-transparent

            px-3

            text-[14px]

            text-white

            outline-none

            placeholder:text-white/28

            font-[
              -apple-system,
              BlinkMacSystemFont,
              'SF Pro Display',
              sans-serif
            ]
          "
        />

        <motion.button

          whileHover={{
            scale: 1.05
          }}

          whileTap={{
            scale: 0.94
          }}

          onClick={sendChat}

          className="
            w-11
            h-11

            rounded-[18px]

            bg-cyan-400

            text-black

            flex
            items-center
            justify-center

            shadow-[0_0_30px_rgba(34,211,238,0.35)]
          "
        >

          <Send size={16} />

        </motion.button>

      </div>

    </div>
  );
}
