; Inno Setup script for the per-user Windows installer.
;
; It installs the app binary and the licence into %LOCALAPPDATA%\flacify\bin, with a Start
; Menu entry and an optional desktop shortcut. The Python sidecar and the weights are
; installed by the app on first run, so this installer is small and needs no admin.

[Setup]
AppId={{6F3B6C2E-9C1A-4F1B-9E3D-2A7C5B8D4E10}
AppName=flacify
AppVersion=0.4.0
AppPublisher=Gabriel Rivero
DefaultDirName={localappdata}\flacify\bin
DefaultGroupName=flacify
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
SourceDir=..
OutputDir=target\release
OutputBaseFilename=flacify-setup
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern
UninstallDisplayIcon={app}\flacify.exe

[Files]
Source: "target\release\flacify.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\flacify"; Filename: "{app}\flacify.exe"
Name: "{autodesktop}\flacify"; Filename: "{app}\flacify.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"
