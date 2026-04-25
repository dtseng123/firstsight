package com.meta.wearable.dat.externalsampleapps.cameraaccess.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.meta.wearable.dat.externalsampleapps.cameraaccess.visionagent.VisionAgentMode

fun visionAgentModeLabel(mode: VisionAgentMode): String =
    when (mode) {
        VisionAgentMode.DIRECT_GEMINI -> "AI Mode: Direct Gemini"
        VisionAgentMode.VISION_AGENT_BACKEND -> "AI Mode: Vision Agent Backend"
    }

@Composable
fun AiModeBadge(
    mode: VisionAgentMode,
    modifier: Modifier = Modifier,
) {
    Text(
        text = visionAgentModeLabel(mode),
        modifier = modifier
            .background(
                color = Color.Black.copy(alpha = 0.72f),
                shape = RoundedCornerShape(999.dp),
            )
            .padding(horizontal = 12.dp, vertical = 8.dp),
        color = Color.White,
        style = MaterialTheme.typography.labelLarge,
    )
}

@Composable
fun AiModeSwitcher(
    mode: VisionAgentMode,
    onModeSelected: (VisionAgentMode) -> Unit,
    modifier: Modifier = Modifier,
) {
    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        AiModeOptionButton(
            label = "Direct Gemini",
            selected = mode == VisionAgentMode.DIRECT_GEMINI,
            onClick = { onModeSelected(VisionAgentMode.DIRECT_GEMINI) },
        )
        AiModeOptionButton(
            label = "Vision Agent Backend",
            selected = mode == VisionAgentMode.VISION_AGENT_BACKEND,
            onClick = { onModeSelected(VisionAgentMode.VISION_AGENT_BACKEND) },
        )
    }
}

@Composable
private fun AiModeOptionButton(
    label: String,
    selected: Boolean,
    onClick: () -> Unit,
) {
    Button(
        onClick = onClick,
        colors = ButtonDefaults.buttonColors(
            containerColor = if (selected) AppColor.DeepBlue else Color.DarkGray,
            contentColor = Color.White,
        ),
    ) {
        Text(label)
    }
}
