import os

CTDDG_ROOT = "/home/muhamed/repo/CTDDG"

log_path = os.path.join(
    CTDDG_ROOT, "outputs", "pretrain", "logs", "log.out"
)

if os.path.exists(log_path):

    with open(log_path) as f:
        lines = f.readlines()

    print(f"Log entries: {len(lines)}")

    if len(lines) > 1:

        print(f"Header: {lines[0].strip()}")
        print(f"Latest: {lines[-1].strip()}")

        if lines[-1].strip() == "Training finished":

            print("\n🎉 Training is COMPLETE!")

        else:

            parts = lines[-1].strip().split('\t')

            if len(parts) >= 2:
                step = int(parts[0])
                print(
                    f"\nProgress: step {step} / 480000 "
                    f"({100 * step / 480000:.1f}%)"
                )

else:
    print("⏳ Training has not started yet (no log file)")
