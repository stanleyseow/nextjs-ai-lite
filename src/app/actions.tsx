'use server';

import { createStreamableValue } from 'ai/rsc';
import { CoreMessage, streamText } from 'ai';
//import { openai } from '@ai-sdk/openai';
import { createOllama } from 'ollama-ai-provider';
import { Weather } from '@/components/weather';
import { generateText } from 'ai';
import { createStreamableUI } from 'ai/rsc';
import { ReactNode } from 'react';
import { z } from 'zod';

const ollama = createOllama({
  // optional settings, e.g.
  baseURL: 'http://192.168.100.156:11434/api',
});

const model = ollama('llama2:7b');


export interface Message {
  role: 'user' | 'assistant';
  content: string;
  display?: ReactNode;
}


// Streaming Chat 
export async function continueTextConversation(messages: CoreMessage[]) {
  const result = await streamText({
    model: model, // using the pre-configured ollama model
    messages,
    provider: ollama, // specify the ollama provider
  });

  const stream = createStreamableValue(result.textStream);
  return stream.value;
}

// Gen UIs 
export async function continueConversation(history: Message[]) {
  const stream = createStreamableUI();

  try {
    const { text, toolResults } = await generateText({
      model: model,
      provider: ollama,
      messages: history.map(msg => ({
        role: msg.role,
        content: msg.content
      })),
      system: 'You are a friendly weather assistant!',
      tools: {
        showWeather: {
          description: 'Show the weather for a given location.',
          parameters: z.object({
            city: z.string().describe('The city to show the weather for.'),
            unit: z.enum(['F']).describe('The unit to display the temperature in'),
          }),
          execute: async ({ city, unit }) => {
            const result = `Here's the weather for ${city}!`;
            stream.done(<Weather city={city} unit={unit} />);
            return result;
          },
        },
      },
    });

    // Ensure we call done() if no tools were used
    if (!toolResults?.length) {
      stream.done(null);
    }

    return {
      messages: [
        ...history,
        {
          role: 'assistant' as const,
          content: text || toolResults?.map(toolResult => toolResult.result).join('') || '',
          display: stream.value,
        },
      ],
    };
  } catch (error) {
    // Ensure stream is completed even on error
    stream.done(null);
    throw error;
  }
}

// Utils
export async function checkAIAvailability() {
  const envVarExists = !!process.env.OPENAI_API_KEY;
  return envVarExists;
}