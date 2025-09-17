import requests
import json
from typing import List, Dict, Optional

class NodeNormalizationClient:
    """Client for interacting with the Node Normalization API and ARAX TRAPI API."""

    def __init__(
        self,
        nn_base_url: str = "https://nodenormalization-sri.renci.org",
        arax_base_url: str = "https://arax.transltr.io/api/arax/v1.4"
    ):
        self.nn_base_url = nn_base_url.rstrip('/')
        self.arax_base_url = arax_base_url.rstrip('/')
        self.nn_session = requests.Session()
        self.arax_session = requests.Session()
    
    def _get_normalized_curie(self, name: str) -> Optional[str]:
        """
        Get normalized CURIE from the ARAX TRAPI API.
        
        Args:
            name: Name to normalize
            
        Returns:
            Normalized CURIE or None on error
        """
        endpoint = f"{self.arax_base_url}/entity"

        try:
            response = self.arax_session.get(
                endpoint,
                params={'q': [name]},
                headers={'Accept': 'application/json'},
                timeout=60
            )
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            print("Request timed out", flush=True)
            return None
        except requests.exceptions.ConnectionError:
            print("Connection error occurred", flush=True)
            return None
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}", flush=True)
            print(f"Response content: {response.text}", flush=True)
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error occurred: {e}", flush=True)
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}", flush=True)
            return None
    
    def get_normalized_node_info(
        self, 
        curie: str = None,
        name: str = None,
        conflate: bool = True,
        drug_chemical_conflate: bool = False,
        description: bool = True
    ) -> Optional[Dict]:
        """
        Get normalized nodes from the API.
        
        Args:
            curie: CURIE to normalize
            name: Name to normalize
            conflate: Whether to conflate nodes
            drug_chemical_conflate: Whether to conflate drugs and chemicals
            description: Whether to include descriptions
            
        Returns:
            Dictionary containing normalized node information or None on error
        """
        
        if not curie and not name:
            print("No CURIE or name provided")
            return None
        
        if name:
            res = self._get_normalized_curie(name)[name]
            if res:
                curie = res.get('id')['identifier']
            else:
                print("No CURIE found", flush=True)
                return None
        
        endpoint = f"{self.nn_base_url}/get_normalized_nodes"
        
        params = {
            'curie': curie,
            'conflate': str(conflate).lower(),
            'drug_chemical_conflate': str(drug_chemical_conflate).lower(),
            'description': str(description).lower()
        }
        
        try:
            response = self.nn_session.get(
                endpoint,
                params=params,
                headers={'Accept': 'application/json'},
                timeout=60
            )
            
            response.raise_for_status()
            return response.json()[curie]
            
        except requests.exceptions.Timeout:
            print("Request timed out", flush=True)
            return None
        except requests.exceptions.ConnectionError:
            print("Connection error occurred", flush=True)
            return None
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}", flush=True)
            print(f"Response content: {response.text}", flush=True)
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error occurred: {e}", flush=True)
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}", flush=True)
            return None

    def get_equivalent_names(self, curie: str = None, name: str = None, **kwargs) -> Optional[List[str]]:
        """Get equivalent names for a single CURIE."""
        result = self.get_normalized_node_info(curie, name, **kwargs)
        
        if not result:
            print("No result found", flush=True)
            return None
        
        eq_ids = result.get('equivalent_identifiers', [])
        names = {x['label'].lower() for x in eq_ids if x.get('label')}
        
        return list(names) if names else None