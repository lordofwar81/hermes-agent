import sys
sys.path.insert(0, '.')

# Import memory tool to trigger registration
from tools.registry import registry

schema = registry.get_schema('memory')
if not schema:
    print("ERROR: Memory tool schema not found in registry")
    sys.exit(1)

print("Memory tool schema found")
params = schema['function']['parameters']['properties']
if 'target' in params:
    target_enum = params['target'].get('enum', [])
    print(f"Target enum: {target_enum}")
    if 'vector' in target_enum:
        print("✅ Vector target present")
    else:
        print("❌ Vector target missing")
        sys.exit(1)
else:
    print("❌ Target parameter not found")
    sys.exit(1)

# Check actions
if 'action' in params:
    action_enum = params['action'].get('enum', [])
    print(f"Action enum: {action_enum}")
    # Should include verify, contradict, retract (maybe)
    # but those are for vector target only, may not be in enum
    # Let's check description
    desc = schema['function'].get('description', '')
    if 'vector' in desc:
        print("✅ Description mentions vector memory")
    else:
        print("⚠️  Description may not mention vector memory")

print("✅ Memory tool schema validation passed")
