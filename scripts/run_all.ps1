# run_all.ps1 - RETIRED (October 2026).
# It ran the arms in a fixed order, with warm-up on the measured rows and unequal Python/JVM warm-up.
# The IEEE Access reviewers asked for randomized order, separate and equal warm-up and run-level analysis.
# Use instead:
#   scripts\resubmission_A_prepare_and_train.ps1   (data + models)
#   scripts\resubmission_B_campaigns.ps1           (all campaigns, overnight)
# or directly:  .venv\Scripts\python.exe scripts\run_campaign.py --campaign <name> --runs 10
Write-Host "run_all.ps1 is retired. Use scripts\resubmission_A_prepare_and_train.ps1, then scripts\resubmission_B_campaigns.ps1." -ForegroundColor Yellow
exit 1
