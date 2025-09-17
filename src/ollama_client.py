"""Ollama client for Hermes 3 and GPT-OSS models."""
import logging
from typing import Dict, Any, List, Union, TYPE_CHECKING
import json
import ollama
from .config import settings

if TYPE_CHECKING:
    from .evaluation_agent import TripleData

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for Ollama-served models (Hermes 4, GPT-OSS)."""
    
    def __init__(self, 
                 model: str = "hermes4:70b",
                 base_url: str = None):
        """Initialize Ollama client.
        
        Args:
            model: Ollama model name (hermes4:70b, gpt-oss:120b, or any available Ollama model)
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = (base_url or settings.ollama_base_url).rstrip('/')
        
        logger.info(f"Initialized Ollama client - Model: {self.model}")
        logger.info(f"Ollama server: {self.base_url}")
        
        # Log model info
        if "hermes4" in model:
            if ":8b" in model:
                logger.info("Using Hermes 4 8B")
            elif ":70b" in model:
                logger.info("Using Hermes 4 70B (Q4_K_XL)")
            elif ":405b" in model:
                logger.info("Using Hermes 4 405B")
        elif "gpt-oss" in model:
            if ":120b" in model:
                logger.info("Using GPT-OSS 120B")
            else:
                logger.info("Using GPT-OSS 20B")

    def _fix_json_formatting(self, json_str: str) -> str:
        """Fix common JSON formatting issues from LLM responses.
        
        Args:
            json_str: Raw JSON string from LLM
            
        Returns:
            Cleaned JSON string with proper formatting
        """
        import re
        
        # Remove any leading/trailing whitespace
        json_str = json_str.strip()
        
        # Handle case where LLM includes explanatory text before/after JSON
        # Look for the actual JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_str, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        
        # Replace single quotes with double quotes for JSON keys and string values
        # Handle keys first
        json_str = re.sub(r"'([^']*)'(\s*:\s*)", r'"\1"\2', json_str)
        
        # Handle string values - be more careful about nested quotes
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        
        # Handle boolean and null values (ensure they're lowercase)
        json_str = re.sub(r'\btrue\b', 'true', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bfalse\b', 'false', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bnull\b', 'null', json_str, flags=re.IGNORECASE)
        
        return json_str

    def _extract_json_manually(self, content: str) -> Dict[str, Any]:
        """Manually extract JSON values when parsing fails.
        
        Args:
            content: Raw LLM response content
            
        Returns:
            Dictionary with extracted values
        """
        import re
        
        # Default values
        result = {
            "is_supported": False,
            "confidence": 0.0,
            "supporting_sentence": None,
            "reasoning": "Manual extraction fallback"
        }
        
        # Extract is_supported
        supported_match = re.search(r'"is_supported":\s*(true|false)', content, re.IGNORECASE)
        if supported_match:
            result["is_supported"] = supported_match.group(1).lower() == 'true'
        
        # Extract confidence
        confidence_match = re.search(r'"confidence":\s*([0-9]*\.?[0-9]+)', content)
        if confidence_match:
            try:
                result["confidence"] = float(confidence_match.group(1))
            except ValueError:
                result["confidence"] = 0.0
        
        # Extract supporting_sentence (handle quotes carefully)
        sentence_match = re.search(r'"supporting_sentence":\s*"([^"]*(?:\\.[^"]*)*)"', content)
        if sentence_match:
            # Unescape any escaped quotes
            sentence = sentence_match.group(1).replace('\\"', '"')
            result["supporting_sentence"] = sentence if sentence.strip() else None
        
        # Extract reasoning (handle quotes carefully)
        reasoning_match = re.search(r'"reasoning":\s*"([^"]*(?:\\.[^"]*)*)"', content)
        if reasoning_match:
            # Unescape any escaped quotes
            reasoning = reasoning_match.group(1).replace('\\"', '"')
            result["reasoning"] = reasoning if reasoning.strip() else "Manual extraction"
        
        return result

    async def generate_response(self, prompt: str) -> Dict[str, Any]:
        """Generate response using Ollama.
        
        Args:
            prompt: Input prompt for the model
            
        Returns:
            Dictionary with response in native Ollama format
        """
        client = ollama.AsyncClient(host=self.base_url, timeout=settings.request_timeout)
        
        try:
            response = await client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1,
                    "top_p": 0.95 if "hermes4" in self.model else 0.9,
                    "top_k": 20 if "hermes4" in self.model else -1,
                    "num_predict": 500
                }
            )
            
            return response
        except ollama.ResponseError as e:
            logger.error(f"Ollama API error: {e}")
            raise Exception(f"Ollama API error: {e}")
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise

    async def evaluate_triple_support(self, triple: Union[List[str], 'TripleData'], abstract: str) -> Dict[str, Any]:
        """Evaluate if an abstract supports a given triple.
        
        Args:
            triple: List [subject, predicate, object] or TripleData object with equivalent names
            abstract: Abstract text to evaluate
            
        Returns:
            Dict with evaluation results
        """
        # Handle both list and TripleData formats
        if hasattr(triple, 'subject'):
            subject = triple.subject
            predicate = triple.predicate
            obj = triple.object
            subject_names = getattr(triple, 'subject_names', [subject])
            object_names = getattr(triple, 'object_names', [obj])
        else:
            subject, predicate, obj = triple
            subject_names = [subject]
            object_names = [obj]
        
        # Use reasoning mode for Hermes 4 for more accurate evaluations
        reasoning_prompt = (
            "You are a deep thinking AI with a strong understanding of medical and biological semantics. Use <think></think> tags to systematically reason through "
            "the evaluation before providing your final JSON response.\n\n"
        )
        
        # Build equivalent names sections
        subject_names_text = ""
        if len(subject_names) > 1:
            subject_names_text = f"\nEquivalent names for '{subject}': {', '.join(subject_names)}"
        
        object_names_text = ""
        if len(object_names) > 1:
            object_names_text = f"\nEquivalent names for '{obj}': {', '.join(object_names)}"
        
        prompt = (
            f"{reasoning_prompt}"
            f"CRITICAL LOGIC RULES:\n"
            f"1. If confidence > 0.8 AND you provide a supporting_sentence, then is_supported MUST be true.\n"
            f"2. If you CANNOT provide a supporting_sentence, set confidence to 0.0 and is_supported MUST be false.\n"
            f"3. Be logically consistent - your confidence, supporting_sentence, and is_supported must align.\n\n"
            f"SEMANTIC UNDERSTANDING RULES:\n"
            f"1. Consider SEMANTIC RELATIONSHIPS, not just exact terminology matches\n"
            f"2. Related medical terms should be considered equivalent (e.g., 'congenital hemiplegia' and 'spastic hemiplegia')\n"
            f"3. For ALL relationship types, recognize synonymous expressions:\n"
            f"   - As an example, the 'subclass_of' relationship can be expressed as 'is_a', 'form of', 'type of', 'category of', 'kind of', 'variant of', 'subset of'\n"
            f"   - Similar principle applies to other relationships (look for equivalent expressions)\n"
            f"4. Medical conditions may be described using synonymous or closely related terms\n"
            f"5. Focus on the CONCEPTUAL relationship described, not exact word matching\n\n"
            f"EVALUATION TASK:\n"
            f"Given the following research abstract, determine if the triple "
            f"'{subject}' {predicate} '{obj}' is supported by the content.\n"
            f"{subject_names_text}"
            f"{object_names_text}\n\n"
            f"Pay special attention to:\n"
            f"- Statements that establish hierarchical relationships (e.g., X is a form/type of Y)\n"
            f"- Related medical terminology that may describe the same or similar concepts\n"
            f"- The equivalent names provided above - any mention of these should be considered as referring to the same concepts\n"
            f"- Implicit relationships that are scientifically established\n\n"
            f"Abstract: {abstract}\n\n"
            f"Respond with ONLY a valid JSON object (use double quotes, not single quotes) containing:\n"
            f"- \"is_supported\": boolean (true if supported, false otherwise)\n"
            f"- \"confidence\": float (0.0 to 1.0, higher confidence means more likely to be supported)\n"
            f"- \"supporting_sentence\": string (the most relevant sentence from the abstract, or null if not supported)\n"
            f"- \"reasoning\": string (brief explanation of your semantic analysis)\n\n"
            f"CRITICAL JSON FORMATTING RULES:\n"
            f"1. Use ONLY double quotes (\") for keys and string values\n"
            f"2. If text contains quotes, escape them with backslash (\\\")\n"
            f"3. Do not include any text before or after the JSON object\n"
            f"4. Ensure all strings are properly closed with matching quotes\n"
            f"5. Use null (not \"null\") for empty supporting_sentence\n\n"
            f"Example: {{\"is_supported\": true, \"confidence\": 0.95, \"supporting_sentence\": \"The study shows X affects Y significantly.\", \"reasoning\": \"Clear causal relationship established.\"}}"
        )

        try:
            response_data = await self.generate_response(prompt)
            content = response_data["message"]["content"]
            
            # Handle Hermes 4 reasoning tags first
            if "</think>" in content:
                # Extract content after </think> tag
                think_end = content.find("</think>") + 8
                content = content[think_end:].strip()
            
            # Extract JSON from response - handle markdown code blocks
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = content[json_start:json_end]
                else:
                    raise ValueError("Could not find valid JSON in LLM response.")
            
            # Clean and parse JSON
            if json_str:
                # Fix common JSON formatting issues from LLM responses
                json_str = self._fix_json_formatting(json_str)
                
                # Try to parse JSON, with fallback handling
                try:
                    evaluation = json.loads(json_str)
                except json.JSONDecodeError as parse_error:
                    # Try to fix common quote issues in JSON strings
                    logger.warning(f"Initial JSON parse failed, attempting to fix quotes: {parse_error}")
                    
                    # More aggressive quote fixing for string values
                    import re
                    # Fix unescaped quotes within string values
                    json_str = re.sub(r'("supporting_sentence":\s*")([^"]*)"([^"]*)"([^"]*")', r'\1\2\\\"\3\\\"\4', json_str)
                    json_str = re.sub(r'("reasoning":\s*")([^"]*)"([^"]*)"([^"]*")', r'\1\2\\\"\3\\\"\4', json_str)
                    
                    try:
                        evaluation = json.loads(json_str)
                        logger.info("Successfully parsed JSON after quote fixing")
                    except json.JSONDecodeError:
                        # Final fallback: try to extract values manually
                        logger.warning("JSON parsing failed completely, attempting manual extraction")
                        evaluation = self._extract_json_manually(content)
            else:
                raise ValueError("Empty JSON content extracted.")
            return evaluation
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {content}. Error: {e}")
            # Try manual extraction as final fallback
            try:
                evaluation = self._extract_json_manually(content)
                logger.info("Successfully extracted JSON manually")
                return evaluation
            except Exception as manual_error:
                logger.error(f"Manual extraction also failed: {manual_error}")
                return {
                    "is_supported": False,
                    "confidence": 0.0,
                    "supporting_sentence": None,
                    "reasoning": f"LLM response not valid JSON: {content[:200]}..."
                }
        except Exception as e:
            logger.error(f"Error during Ollama evaluation: {e}")
            raise
