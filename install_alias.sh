#!/bin/bash
# Add MAYA to PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHELL_RC=""

if [[ "$SHELL" == *"zsh"* ]]
  SHELL_RC="$HOME/.zshrc"
elif [[ "$SHELL" == *"bash"* ]]
  SHELL_RC="$HOME/.bashrc"
fi

if [ -n "$SHELL_RC" ]; then
    echo "export PATH=\"$SCRIPT_DIR:\$PATH\"" >> "$SHELL_RC"
    echo "alias maya=\"$SCRIPT_DIR/maya\"" >> "$SHELL_RC"
    echo "MAYA installed! Restart your terminal or run: source $SHELL_RC"
else
    echo "Add to your shell config:"
    echo "  export PATH=\"$SCRIPT_DIR:\$PATH\""
    echo "  alias maya=\"$SCRIPT_DIR/maya\""
fi
