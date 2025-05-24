#!/usr/bin/env python3
"""
Unified Application Launcher for Marker
Provides a single entry point for all Marker applications and services.
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import Optional

class MarkerAppLauncher:
    """Unified launcher for all Marker applications"""
    
    def __init__(self):
        self.apps = {
            "server": {
                "description": "Launch Marker MCP API server (MCP-compliant)",
                "module": "marker.mcp_server",
                "function": "main"
            },
            "convert": {
                "description": "Convert PDFs using unified conversion system",
                "module": "unified_conversion",
                "function": "main"
            },
            "monitor": {
                "description": "Launch monitoring system",
                "module": "unified_monitoring",
                "function": "main"
            }
        }
    
    def list_apps(self):
        """List all available applications"""
        print("\n🚀 Available Marker Applications:")
        print("=" * 50)
        for app_name, app_info in self.apps.items():
            print(f"  {app_name:12} - {app_info['description']}")
        print("   Example: python marker_launcher.py server")
        print("   Example: python marker_launcher.py convert --help")
        print("   Example: python marker_launcher.py monitor --status")
    
    def launch_app(self, app_name: str, app_args: Optional[list] = None):
        """Launch a specific application"""
        if app_name not in self.apps:
            print(f"❌ Unknown application: {app_name}")
            print(f"❌ Unknown application: {app_name}")
            self.list_apps()
            return False
        
        app_info = self.apps[app_name]
        app_args = app_args or []
        
        try:
            print(f"🚀 Launching {app_name}: {app_info['description']}")
            
            # Dynamic import and execution
            module = __import__(app_info['module'], fromlist=[app_info['function']])
            app_function = getattr(module, app_info['function'])
            
            # Modify sys.argv for the app
            original_argv = sys.argv.copy()
            sys.argv = [app_info['module']] + app_args
            
            try:
                app_function()
            finally:
                sys.argv = original_argv
            
            return True
            
        except ImportError as e:
            print(f"❌ Could not import {app_info['module']}: {e}")
            return False
        except Exception as e:
            print(f"❌ Error launching {app_name}: {e}")
            return False
    
    def interactive_launcher(self):
        """Interactive application launcher"""
        print("\n🎯 Marker Interactive Launcher")
        print("=" * 40)
        
        while True:
            print(f"\nAvailable applications:")
            for i, (app_name, app_info) in enumerate(self.apps.items(), 1):
                print(f"  {i}. {app_name} - {app_info['description']}")
            
            print(f"  0. Exit")
            
            try:
                choice = input(f"\nSelect application (0-{len(self.apps)}): ").strip()
                
                if choice == "0":
                    print("👋 Goodbye!")
                    break
                
                try:
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(self.apps):
                        app_name = list(self.apps.keys())[choice_num - 1]
                        self.launch_app(app_name)
                    else:
                        print("❌ Invalid choice. Please try again.")
                except ValueError:
                    # Try direct app name
                    if choice in self.apps:
                        self.launch_app(choice)
                    else:
                        print("❌ Invalid input. Please enter a number or app name.")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Unified Marker Application Launcher (MCP-compliant)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Applications:
  server       - Launch Marker MCP API server (MCP-compliant)
  convert      - Convert PDFs using unified conversion system
  monitor      - Launch monitoring system

Examples:
  python marker_launcher.py server
  python marker_launcher.py convert --help
  python marker_launcher.py monitor --status
        """
    )
    
    parser.add_argument("app", nargs="?", help="Application to launch")
    parser.add_argument("--list", action="store_true", help="List all available applications")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive mode")
    
    # Parse known args to allow passing arguments to apps
    args, unknown_args = parser.parse_known_args()
    
    launcher = MarkerAppLauncher()
    
    if args.list:
        launcher.list_apps()
    elif args.interactive:
        launcher.interactive_launcher()
    elif args.app:
        success = launcher.launch_app(args.app, unknown_args)
        if not success:
            sys.exit(1)
    else:
        launcher.list_apps()

if __name__ == "__main__":
    main()
