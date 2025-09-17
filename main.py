#!/usr/bin/env python3
"""Main interface for triple checking."""
import argparse
import asyncio
import logging
import sys
from src.triple_evaluator import TripleEvaluatorSystem
from src.node_normalization import NodeNormalizationClient

def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Check research triples against PMID abstracts using Ollama LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using names (requires node normalization)
  python main.py --model gpt-oss --triple_name "SIX1" "affects" "Cell Proliferation" --pmids 16186693 29083299
  python main.py --model hermes4 --triple_name "SIX1" "affects" "Cell Proliferation" --pmids 16186693 29083299
  
  # Using CURIEs directly
  python main.py --model gpt-oss --triple_curie "NCBIGene:6495" "affects" "UMLS:C0596290" --pmids 16186693 29083299
        """
    )
    
    # Triple specification - mutually exclusive options
    triple_group = parser.add_mutually_exclusive_group(required=True)
    triple_group.add_argument(
        '--triple_curie', 
        nargs=3,
        metavar=('SUBJECT_CURIE', 'PREDICATE', 'OBJECT_CURIE'),
        help='Research triple as CURIEs (e.g., "NCBIGene:6495" "affects" "UMLS:C0596290")'
    )
    triple_group.add_argument(
        '--triple_name', 
        nargs=3,
        metavar=('SUBJECT_NAME', 'PREDICATE', 'OBJECT_NAME'),
        help='Research triple as names (e.g., "SIX1" "affects" "Cell Proliferation")'
    )
    
    # PMID specification
    pmid_group = parser.add_mutually_exclusive_group(required=True)
    pmid_group.add_argument(
        '--pmids',
        nargs='+',
        help='List of PMIDs to evaluate'
    )
    pmid_group.add_argument(
        '--pmids-file',
        help='File containing PMIDs (one per line)'
    )
    
    # Model selection (required)
    parser.add_argument('--model', 
                               choices=['hermes4', 'gpt-oss'],
        help='LLM provider to use (hermes4 or gpt-oss)',
                       required=True)
    
    # Optional arguments
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Parse triple and get equivalent names
    normalization_client = NodeNormalizationClient()
    
    if args.triple_curie:
        subject_curie, predicate, object_curie = args.triple_curie
        print(f"Getting equivalent names for CURIEs: {subject_curie}, {object_curie}")
        
        # Get equivalent names for subject and object
        subject_names = normalization_client.get_equivalent_names(curie=subject_curie)
        object_names = normalization_client.get_equivalent_names(curie=object_curie)
        
        if not subject_names:
            print(f"Warning: No equivalent names found for subject CURIE: {subject_curie}", file=sys.stderr)
            subject_names = [subject_curie]
        if not object_names:
            print(f"Warning: No equivalent names found for object CURIE: {object_curie}", file=sys.stderr)
            object_names = [object_curie]
            
        triple = [subject_names[0], predicate, object_names[0]]  # Use primary name for display
        triple_with_names = {
            'subject': subject_names[0],
            'predicate': predicate,
            'object': object_names[0],
            'subject_names': subject_names,
            'object_names': object_names
        }
        
    elif args.triple_name:
        subject_name, predicate, object_name = args.triple_name
        subject_name = subject_name.replace(',','')
        object_name = object_name.replace(',','')
        print(f"Getting equivalent names for names: {subject_name}, {object_name}")
        
        # Get equivalent names for subject and object
        subject_names = normalization_client.get_equivalent_names(name=subject_name)
        object_names = normalization_client.get_equivalent_names(name=object_name)
        
        if not subject_names:
            print(f"Warning: No equivalent names found for subject name: {subject_name}", file=sys.stderr)
            subject_names = [subject_name]
        if not object_names:
            print(f"Warning: No equivalent names found for object name: {object_name}", file=sys.stderr)
            object_names = [object_name]
            
        triple = [subject_name, predicate, object_name]  # Use original name for display
        triple_with_names = {
            'subject': subject_name,
            'predicate': predicate,
            'object': object_name,
            'subject_names': subject_names,
            'object_names': object_names
        }
    else:
        print("Error: Either --triple_curie or --triple_name must be provided", file=sys.stderr)
        return 1
    
    # Parse PMIDs
    if args.pmids:
        pmids = args.pmids
    else:
        try:
            with open(args.pmids_file, 'r') as f:
                pmids = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"PMIDs file not found: {args.pmids_file}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error reading PMIDs file: {e}", file=sys.stderr)
            return 1
    
    if not pmids:
        print("No PMIDs provided", file=sys.stderr)
        return 1
    
    print(f"Checking triple {triple} against {len(pmids)} PMIDs...")
    print("=" * 60)
    
    try:
        # Create evaluator system with specified model
        evaluator = TripleEvaluatorSystem(
            llm_provider=args.model
        )
        
        # Run the evaluation with enriched triple data
        results = await evaluator.evaluate_triple_with_names(
            subject=triple_with_names['subject'],
            predicate=triple_with_names['predicate'], 
            object_=triple_with_names['object'],
            subject_names=triple_with_names['subject_names'],
            object_names=triple_with_names['object_names'],
            pmids=pmids
        )
        
        # Output results
        formatted_output = results.format_output(verbose=args.verbose) if hasattr(results, 'format_output') else str(results)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(formatted_output)
            print(f"Results written to {args.output}")
        else:
            print(formatted_output)
        
        # Print final summary
        summary = results.get_summary()
        print("\n" + "=" * 60)
        print("CHECK SUMMARY")
        print("=" * 60)
        print(f"Total PMIDs: {summary['total_pmids']}")
        print(f"Supported: {summary['supported_pmids']} ({summary['supported_percentage']}%)")
        print(f"Not Supported: {summary['unsupported_pmids']} ({summary['unsupported_percentage']}%)")
        
        return 0
        
    except Exception as e:
        print(f"Error during check: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))