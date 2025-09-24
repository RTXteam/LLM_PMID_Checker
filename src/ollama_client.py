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
            "evidence_category": "dont_know",
            "supporting_sentence": None,
            "reasoning": "Manual extraction fallback",
            "subject_mentioned": False,
            "object_mentioned": False
        }
        
        # Extract is_supported
        supported_match = re.search(r'"is_supported":\s*(true|false)', content, re.IGNORECASE)
        if supported_match:
            result["is_supported"] = supported_match.group(1).lower() == 'true'
        
        # Extract evidence_category
        category_match = re.search(r'"evidence_category":\s*"([^"]*)"', content)
        if category_match:
            result["evidence_category"] = category_match.group(1)
        
        
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
        
        # Extract subject_mentioned
        subject_mentioned_match = re.search(r'"subject_mentioned":\s*(true|false)', content, re.IGNORECASE)
        if subject_mentioned_match:
            result["subject_mentioned"] = subject_mentioned_match.group(1).lower() == 'true'
        
        # Extract object_mentioned
        object_mentioned_match = re.search(r'"object_mentioned":\s*(true|false)', content, re.IGNORECASE)
        if object_mentioned_match:
            result["object_mentioned"] = object_mentioned_match.group(1).lower() == 'true'
        
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
                    "num_predict": 800
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
            # Get qualifier information
            qualified_predicate = getattr(triple, 'qualified_predicate', None)
            qualified_object_aspect = getattr(triple, 'qualified_object_aspect', None)
            qualified_object_direction = getattr(triple, 'qualified_object_direction', None)
            has_qualifiers = getattr(triple, 'has_qualifiers', lambda: False)()
        else:
            subject, predicate, obj = triple
            subject_names = [subject]
            object_names = [obj]
            qualified_predicate = None
            qualified_object_aspect = None
            qualified_object_direction = None
            has_qualifiers = False
        
        # Use model-specific prompting approaches
        if "gpt-oss" in self.model.lower():
            # GPT-OSS uses thinking field for reasoning, so we need explicit instructions for final output
            reasoning_prompt = (
                "You are a deep thinking AI with a strong understanding of medical and biological semantics. Analyze the abstract carefully and provide your final answer as a JSON object.\n\n"
                "IMPORTANT: After your analysis, you MUST provide a final JSON response in this exact format:\n"
                "FINAL_ANSWER: {\"is_supported\": true/false, \"supporting_sentence\": \"quote\" or null, \"reasoning\": \"brief explanation\"}\n\n"
            )
        else:
            # Hermes 4 and other models use <think></think> tags
            reasoning_prompt = (
                "You are a deep thinking AI with a strong understanding of medical and biological semantics. Analyze the abstract carefully and provide your final answer as a JSON object.\n\n"
                "Use <think></think> tags to systematically reason through the evaluation before providing your final JSON response.\n\n"
            )
        
        # Build equivalent names sections
        subject_names_text = ""
        if len(subject_names) > 1:
            subject_names_text = f"\nEquivalent names for '{subject}': {', '.join(subject_names)}"
        
        object_names_text = ""
        if len(object_names) > 1:
            object_names_text = f"\nEquivalent names for '{obj}': {', '.join(object_names)}"
        
        # Build the triple description based on whether qualifiers are present
        if has_qualifiers:
            # Build qualified description
            qualified_parts = []
            if qualified_object_direction:
                qualified_parts.append(qualified_object_direction)
            if qualified_object_aspect:
                qualified_parts.append(qualified_object_aspect)
            
            qualified_description = " ".join(qualified_parts)
            triple_description = f"'{subject}' {qualified_predicate} {qualified_description} of '{obj}'"
            
            # Add qualifier-specific guidance
            qualifier_guidance = (
                f"CRITICAL QUALIFIER EVALUATION:\n"
                f"The relationship is qualified with:\n"
                f"- Qualified predicate: {qualified_predicate}\n"
                f"- Object aspect: {qualified_object_aspect or 'not specified'}\n"
                f"- Direction: {qualified_object_direction or 'not specified'}\n\n"
                f"STRICT REQUIREMENTS - ALL must be present for support:\n"
                f"1. CAUSAL RELATIONSHIP: Must show '{qualified_predicate}' (not just correlation, association, or involvement)\n"
                f"2. DIRECTION: Must explicitly show '{qualified_object_direction}' (not opposite direction or unclear)\n"
                f"3. ASPECT: Must relate to '{qualified_object_aspect}' (activity/function OR abundance/levels)\n\n"
                f"EVIDENCE STANDARDS:\n"
                f"- 'Functions in' or 'involved in' = RELATED BUT NOT DIRECT (insufficient for causation)\n"
                f"- Terms matching the requested direction '{qualified_object_direction}' = DIRECT EVIDENCE\n"
                f"- Terms opposite to the requested direction '{qualified_object_direction}' = OPPOSITE ASSERTION\n"
                f"- General involvement without clear direction = RELATED BUT NOT DIRECT\n\n"
                f"DIRECTION MATCHING EXAMPLES:\n"
                f"- If requesting 'increased': 'promotes', 'enhances', 'stimulates', 'increases', 'upregulates' = DIRECT EVIDENCE\n"
                f"- If requesting 'increased': 'inhibits', 'reduces', 'decreases', 'suppresses', 'downregulates' = OPPOSITE ASSERTION\n"
                f"- If requesting 'decreased': 'inhibits', 'reduces', 'decreases', 'suppresses', 'downregulates' = DIRECT EVIDENCE\n"
                f"- If requesting 'decreased': 'promotes', 'enhances', 'stimulates', 'increases', 'upregulates' = OPPOSITE ASSERTION\n\n"
            )
        else:
            triple_description = f"'{subject}' {predicate} '{obj}'"
            qualifier_guidance = ""
        
        prompt = (
            f"{reasoning_prompt}"
            f"CRITICAL LOGIC RULES:\n"
            f"1. If you CANNOT provide a supporting_sentence, is_supported MUST be false.\n"
            f"2. Be logically consistent - your supporting_sentence and is_supported must align.\n\n"
            f"SEMANTIC UNDERSTANDING RULES:\n"
            f"1. Consider SEMANTIC RELATIONSHIPS, not just exact terminology matches\n"
            f"2. Related medical terms should be considered equivalent (e.g., 'congenital hemiplegia' and 'spastic hemiplegia')\n"
            f"3. For ALL relationship types, recognize synonymous expressions:\n"
            f"   - As an example, the 'subclass_of' relationship can be expressed as 'is_a', 'form of', 'type of', 'category of', 'kind of', 'variant of', 'subset of'\n"
            f"   - Similar principle applies to other relationships (look for equivalent expressions)\n"
            f"4. Medical conditions may be described using synonymous or closely related terms\n"
            f"5. Focus on the CONCEPTUAL relationship described, not exact word matching\n\n"
            f"{qualifier_guidance}"
            f"EVALUATION TASK:\n"
            f"Given the following research abstract, determine if the triple "
            f"{triple_description} is supported by the content.\n"
            f"{subject_names_text}"
            f"{object_names_text}\n\n"
            f"Pay special attention to:\n"
            f"- Statements that establish hierarchical relationships (e.g., X is a form/type of Y)\n"
            f"- Related medical terminology that may describe the same or similar concepts\n"
            f"- The equivalent names provided above - any mention of these should be considered as referring to the same concepts\n"
            f"- Implicit relationships that are scientifically established\n\n"
            f"Abstract: {abstract}\n\n"
            f"Respond with ONLY a valid JSON object (use double quotes, not single quotes) containing:\n"
            f"- \"is_supported\": boolean (true only if ALL qualifier requirements are met)\n"
            f"- \"evidence_category\": string (one of: \"direct_evidence\", \"opposite_assertion\", \"related_not_direct\", \"not_supported\", \"dont_know\")\n"
            f"- \"supporting_sentence\": string (the most relevant sentence from the abstract, or null if not supported)\n"
            f"- \"reasoning\": string (detailed explanation including qualifier assessment)\n"
            f"- \"subject_mentioned\": boolean (true if any equivalent name of the subject '{subject}' is mentioned in the abstract)\n"
            f"- \"object_mentioned\": boolean (true if any equivalent name of the object '{obj}' is mentioned in the abstract)\n\n"
            f"EVIDENCE CATEGORY DEFINITIONS:\n"
            f"- \"direct_evidence\": Clear evidence supporting ALL aspects (causation + direction + aspect)\n"
            f"- \"opposite_assertion\": Evidence showing opposite direction or contradictory relationship\n"
            f"- \"related_not_direct\": Related to the topic but missing qualifier specificity\n"
            f"- \"not_supported\": No relevant evidence found\n"
            f"- \"dont_know\": Ambiguous or insufficient information to categorize\n\n"
            f"CRITICAL JSON FORMATTING RULES:\n"
            f"1. Use ONLY double quotes (\") for keys and string values\n"
            f"2. If text contains quotes, escape them with backslash (\\\")\n"
            f"3. Do not include any text before or after the JSON object\n"
            f"4. Ensure all strings are properly closed with matching quotes\n"
            f"5. Use null (not \"null\") for empty supporting_sentence\n\n"
            f"Example: {{\"is_supported\": true, \"supporting_sentence\": \"The study shows X affects Y significantly.\", \"reasoning\": \"Clear causal relationship established.\"}}"
        )

        try:
            response_data = await self.generate_response(prompt)
            
            # Handle both Pydantic objects (newer ollama) and dict format (older versions)
            if hasattr(response_data, 'message'):
                # Pydantic object format
                message = response_data.message
                content = message.content if hasattr(message, 'content') and message.content is not None else ""
                thinking_content = message.thinking if hasattr(message, 'thinking') and message.thinking is not None else ""
            else:
                # Dictionary format (legacy)
                message = response_data["message"]
                content = message.get("content", "") or ""
                thinking_content = message.get("thinking", "") or ""
            
            logger.debug(f"Content field: {content[:200]}...")
            logger.debug(f"Thinking field: {thinking_content[:200]}...")
            
            # Handle FINAL_ANSWER: pattern in content (for GPT-OSS)
            if "FINAL_ANSWER:" in content:
                # Extract JSON after FINAL_ANSWER:
                answer_start = content.find("FINAL_ANSWER:") + 13
                json_part = content[answer_start:].strip()
                content = json_part
                logger.info("Found FINAL_ANSWER pattern in content field, extracted JSON")
            
            # Fallback: Handle models that put response in thinking field (legacy support)
            elif not content.strip() and thinking_content:
                logger.info("Using thinking field as model put response there instead of content field")
                
                # Look for FINAL_ANSWER: pattern in thinking content
                if "FINAL_ANSWER:" in thinking_content:
                    # Extract JSON after FINAL_ANSWER:
                    answer_start = thinking_content.find("FINAL_ANSWER:") + 13
                    json_part = thinking_content[answer_start:].strip()
                    content = json_part
                    logger.info("Found FINAL_ANSWER pattern in thinking field")
                else:
                    # Fallback: look for any JSON-like content in the thinking field
                    content = thinking_content
                    logger.debug(f"Thinking field content (first 500 chars): {thinking_content[:500]}")
                    
                    # If no JSON found in thinking, try to extract the decision and create JSON
                    if '{' not in thinking_content:
                        logger.warning("No JSON found in thinking field, attempting to create JSON from reasoning")
                        # Try to determine if it's supported based on keywords in the reasoning
                        is_supported = any(keyword in thinking_content.lower() for keyword in [
                            'supported', 'supports', 'evidence', 'indicates', 'suggests', 'shows', 'demonstrates'
                        ])
                        # Create a basic JSON response
                        content = f'{{"is_supported": {str(is_supported).lower()}, "supporting_sentence": null, "reasoning": "Based on model reasoning in thinking field"}}'
                        logger.info(f"Created JSON from reasoning: {content}")
            
            # New fallback: If content is empty but we have thinking, try to use the thinking content directly
            elif not content.strip() and not thinking_content.strip():
                logger.error("Both content and thinking fields are empty")
                raise ValueError("Model returned empty response in both content and thinking fields")
            
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
                # Look for JSON object in content - handle multiline JSON
                import re
                # Find the first { and the last } that would complete a valid JSON object
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Fallback to simple brace matching
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start and json_end != 0:
                        json_str = content[json_start:json_end]
                    else:
                        logger.error(f"Could not find JSON braces in content. Content: {content[:1000]}")
                        # If JSON is incomplete, try manual extraction
                        logger.warning("Attempting manual extraction from incomplete JSON")
                        evaluation = self._extract_json_manually(content)
                        return evaluation
            
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
                    "supporting_sentence": None,
                    "reasoning": f"LLM response not valid JSON: {content[:200]}..."
                }
        except Exception as e:
            logger.error(f"Error during Ollama evaluation: {e}")
            raise
