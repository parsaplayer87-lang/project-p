import time

import main


TARGETS = (
    "https://zoomit.ir/business/471017-jensen-huang-resigns-nvidia-ceo?thread=10439268&feedbackId=10440800#comments",
    "https://zoomit.ir/mobile/471042-samsung-69-percent-foldable-market-share-galaxy-z-fold8?thread=10444404&feedbackId=10444404#comments",
    "https://zoomit.ir/business/470936-nvidia-jensen-huang-china-investigation?thread=10437642&feedbackId=10444944#comments",
    "https://zoomit.ir/business/471017-jensen-huang-resigns-nvidia-ceo?thread=10439268&feedbackId=10445504#comments",
    "https://zoomit.ir/biology/470978-why-hammerhead-sharks-have-hammer-shaped-heads?thread=10447237&feedbackId=10447237#comments",
)
EXPECTED_BOT_ID = "8da8e1d4-bb7b-42b3-8680-7dc57eb4aac6"


def run():
    main.STATE.load()
    if main.DRY_RUN or not main.STATE.fernet or not main.MEM_TOKEN:
        raise RuntimeError("Recovery requires live posting and both encrypted and remote memory")
    main.zoomit_token()
    if main.BOT_USER_ID != EXPECTED_BOT_ID:
        raise RuntimeError("Unexpected bot account; no replies attempted")
    pending = list(TARGETS)
    deadline = time.monotonic() + 45 * 60
    try:
        while pending and time.monotonic() < deadline:
            for url in pending[:]:
                feedback_id = main.parse_mention_link(url)["feedback_id"]
                marker = "manual-recovery-20261114-" + feedback_id
                if marker in main.STATE.data["processed"]:
                    main.log.info("RECOVERY %s: previously completed", feedback_id)
                    pending.remove(url)
                    continue
                outcome = main.handle_mention({"id": marker, "link": {"url": url}})
                if outcome == "done":
                    main.STATE.mark_processed(marker)
                    pending.remove(url)
                    main.log.info("RECOVERY %s: completed through main.handle_mention", feedback_id)
                else:
                    attempts = main.STATE.data["attempts"].get(marker, 0) + 1
                    main.STATE.data["attempts"][marker] = attempts
                    main.STATE.dirty = True
                    main.log.info("RECOVERY %s: pending after attempt %d", feedback_id, attempts)
                main.STATE.save_if_dirty()
                if time.monotonic() >= deadline:
                    break
                time.sleep(main.AI_MIN_GAP_SEC)
            if pending and time.monotonic() < deadline:
                time.sleep(min(main.CHECK_INTERVAL_SEC, max(0, deadline - time.monotonic())))
        if pending:
            ids = [main.parse_mention_link(url)["feedback_id"] for url in pending]
            raise RuntimeError("Recovery window ended with pending feedback IDs: " + ", ".join(ids))
        main.log.info("RECOVERY complete: all five targets handled; existing history retained")
    finally:
        main.STATE.save()


if __name__ == "__main__":
    run()
