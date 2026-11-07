# Read-only evidence for "llama-server.exe vanished or became inaccessible" on Windows.
# Changes no settings, installs nothing, touches no antivirus configuration.
# Usage: powershell -File scripts\windows_runtime_evidence.ps1 [-Path <llama-server.exe>] [-Hours 24]
# Adapted from the exewatch diagnostics (External R&D, 2026-10-23). Not yet run on a native Windows host.
param(
    [string]$Path = "$env:LOCALAPPDATA\arc-llama\runtime\llama-server.exe",
    [int]$Hours = 24
)
$ErrorActionPreference = 'SilentlyContinue'
$since = (Get-Date).AddHours(-$Hours)

"== Target: $Path"
if (Test-Path -LiteralPath $Path) {
    Get-Item -LiteralPath $Path -Force | Select-Object FullName, Length, Attributes, CreationTime, LastWriteTime | Format-List
    "-- Sibling files (an exe gone while its DLLs remain points at a targeted removal or rename)"
    Get-ChildItem -LiteralPath (Split-Path -Parent $Path) -Force | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
    Get-Acl -LiteralPath $Path | Format-List
    Get-AuthenticodeSignature -LiteralPath $Path | Select-Object Status, SignerCertificate
    Get-Item -LiteralPath $Path -Stream * | Select-Object Stream, Length
    Get-Content -LiteralPath $Path -Stream Zone.Identifier
} else {
    "MISSING at the time of this check"
}

"== Defender events (1116 detected, 1117 action taken, 1118/1119 action failed)"
Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-Windows Defender/Operational'; Id=1116,1117,1118,1119; StartTime=$since} |
    Select-Object TimeCreated, Id, Message | Format-List
"== Defender threat history"
Get-MpThreatDetection | Select-Object InitialDetectionTime, ThreatID, ActionSuccess, Resources
Get-MpThreat | Select-Object ThreatID, ThreatName, IsActive, Resources
"== Defender state"
Get-MpComputerStatus | Select-Object AMRunningMode, RealTimeProtectionEnabled, IsTamperProtected, AntivirusSignatureVersion

"== Code integrity, Smart App Control and AppLocker blocks"
Get-WinEvent -LogName 'Microsoft-Windows-CodeIntegrity/Operational' -MaxEvents 50 | Select-Object TimeCreated, Id, Message
Get-WinEvent -LogName 'Microsoft-Windows-AppLocker/EXE and DLL' -MaxEvents 50 | Select-Object TimeCreated, Id, Message

"== Sysmon file deletes (only if Sysmon is already installed)"
Get-WinEvent -FilterHashtable @{LogName='Microsoft-Windows-Sysmon/Operational'; Id=23,26} -MaxEvents 100 |
    Select-Object TimeCreated, Id, Message

"== Storage Sense temp cleanup policy"
Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\CurrentVersion\StorageSense\Parameters\StoragePolicy'
