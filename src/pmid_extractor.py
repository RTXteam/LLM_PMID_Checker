"""PMID abstract extraction using easy-entrez."""
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from easy_entrez import EntrezAPI

logger = logging.getLogger(__name__)

@dataclass
class AbstractData:
    """Container for PMID abstract data."""
    pmid: str
    title: str
    abstract: str
    error: Optional[str] = None

class PMIDExtractor:
    """Extracts abstracts from PubMed using easy-entrez."""
    
    def __init__(self, api_key, email):
        """Initialize the PMID extractor.
        
        Args:
            api_key: NCBI E-utilities API key for higher rate limits
            email: Email for NCBI requests (required for higher volumes)
        """
        # Initialize EntrezAPI
        self.entrez_api = EntrezAPI(
            tool="llm_pmid_checker",
            email=email,
            api_key=api_key,
            return_type='xml'
        )
        
    def extract_abstracts(self, pmids: List[str]) -> Dict[str, AbstractData]:
        """Extract abstracts for a list of PMIDs.
        
        Args:
            pmids: List of PubMed identifiers
            
        Returns:
            Dictionary mapping PMID to AbstractData
        """
        results = {}
        
        if not pmids:
            return results
        
        try:
            # Fetch articles from PubMed
            response = self.entrez_api.fetch(
                pmids,
                max_results=len(pmids),
                database='pubmed'
            )
            
            # Parse the XML response data
            if response.data:
                import xml.etree.ElementTree as ET
                root = response.data if hasattr(response.data, 'tag') else ET.fromstring(str(response.data))
                
                # Find all PubmedArticle elements
                articles = root.findall('.//PubmedArticle')
                
                for article in articles:
                    try:
                        # Extract PMID
                        pmid_elem = article.find('.//PMID')
                        pmid = pmid_elem.text if pmid_elem is not None else ""
                        
                        if not pmid:
                            continue
                        
                        # Extract title - use itertext() to handle nested formatting tags
                        title_elem = article.find('.//ArticleTitle')
                        title = ''.join(title_elem.itertext()).strip() if title_elem is not None else ""
                        
                        # Extract abstract - combine all AbstractText elements
                        abstract_elems = article.findall('.//AbstractText')
                        abstract_parts = []
                        for elem in abstract_elems:
                            text = ''.join(elem.itertext()).strip()
                            if text:
                                # Check if there's a label attribute
                                label = elem.get('Label', '')
                                if label:
                                    abstract_parts.append(f"{label}: {text}")
                                else:
                                    abstract_parts.append(text)
                        
                        abstract = ' '.join(abstract_parts) if abstract_parts else ""
                        
                        results[pmid] = AbstractData(
                            pmid=pmid,
                            title=title,
                            abstract=abstract,
                            error=None if abstract else "No abstract available"
                        )
                        
                    except Exception as e:
                        logger.error(f"Error parsing article data: {e}")
                        continue
            
            # Handle PMIDs that weren't found in the response
            for pmid in pmids:
                if pmid not in results:
                    results[pmid] = AbstractData(
                        pmid=pmid,
                        title="",
                        abstract="",
                        error="PMID not found or could not be retrieved"
                    )
                    
        except Exception as e:
            logger.error(f"Failed to extract abstracts: {e}")
            for pmid in pmids:
                results[pmid] = AbstractData(
                    pmid=pmid,
                    title="",
                    abstract="",
                    error=f"Extraction failed: {str(e)}"
                )
        
        return results
    